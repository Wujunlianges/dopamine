import os
import numpy as np
import tensorflow as tf
from dopamine.discrete_domains import gym_lib
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment, atari_lib
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf
# import trfl

from dopamine.discrete_domains.pg_networks import ActorNetwork


def discount_reward(rewards, gamma=0.99, is_norm=True):
    discount_rewards = np.zeros_like(rewards)
    running_add = 0
    for i in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[i]
        discount_rewards[i] = running_add
    if is_norm:
        discount_rewards = (discount_rewards - discount_rewards.mean()) / (
            1e-9 + discount_rewards.std())
    return discount_rewards


def compute_advantages(discount,
                       gae_lambda,
                       max_len,
                       baselines,
                       rewards,
                       name=None):
    with tf.name_scope(name, 'compute_advantages',
                       [discount, gae_lambda, max_len, baselines, rewards]):
        gamma_lambda = tf.constant(float(discount) * float(gae_lambda),
                                   dtype=tf.float32,
                                   shape=[max_len, 1, 1])
        advantage_filter = tf.compat.v1.cumprod(gamma_lambda, exclusive=True)

        pad = tf.zeros_like(baselines[:, :1])
        baseline_shift = tf.concat([baselines[:, 1:], pad], 1)
        deltas = rewards + discount * baseline_shift - baselines
        deltas_pad = tf.expand_dims(tf.concat(
            [deltas, tf.zeros_like(deltas[:, :-1])], axis=1),
                                    axis=2)
        adv = tf.nn.conv1d(deltas_pad,
                           advantage_filter,
                           stride=1,
                           padding='VALID')
        advantages = tf.reshape(adv, [-1])
    return advantages


@gin.configurable
class NPGAgent:
    def __init__(self,
                 sess,
                 num_actions,
                 actor_dist_fn=tf.distributions.Categorical,
                 observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=atari_lib.NATURE_DQN_DTYPE,
                 actor_network=ActorNetwork,
                 gamma=0.99,
                 update_horizon=1,
                 actor_out_fn=tf.nn.softmax,
                 tf_device='/cpu:*',
                 eval_mode=False,
                 use_staging=False,
                 max_tf_checkpoints_to_keep=4,
                 actor_optimizer=tf.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     momentum=0.0,
                     epsilon=0.00001,
                     centered=True),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 allow_partial_reload=False):
        assert isinstance(observation_shape, tuple)
        tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t update_horizon: %f', update_horizon)
        tf.logging.info('\t tf_device: %s', tf_device)
        tf.logging.info('\t actor_optimizer: %s', actor_optimizer)
        tf.logging.info('\t max_tf_checkpoints_to_keep: %d',
                        max_tf_checkpoints_to_keep)

        self._sess = sess
        self._last_action = np.random.randint(num_actions)
        self._last_observation = None
        self._observation = None

        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = observation_dtype
        self.gamma = gamma
        self.training_steps = 0
        self.update_horizon = update_horizon
        self.actor_optimizer = actor_optimizer

        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency
        self.allow_partial_reload = allow_partial_reload

        self.actor_dist_fn = actor_dist_fn
        self.actor_out_fn = actor_out_fn
        self.actor_network = actor_network
        self.eval_mode = eval_mode

        self.rewards = []
        self.actions = []
        self.states = []
        self.is_terminal = []

        with tf.device(tf_device):
            state_shape = (1, ) + self.observation_shape
            states_shape = (None, ) + self.observation_shape
            self.state = np.zeros(state_shape)
            self.actions_ph = tf.placeholder(tf.int32, (None, ),
                                             name='actions_ph')
            self.rewards_ph = tf.placeholder(tf.float32, (None, ),
                                             name='rewards_ph')
            self.states_ph = tf.placeholder(self.observation_dtype,
                                            states_shape,
                                            name='states_ph')
            self.is_terminal_ph = tf.placeholder(tf.float32, (None, ),
                                                 name='is_terminal_ph')
            self._build_networks()
            self._train_op = self._build_train_op()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()

        var_map = atari_lib.maybe_transform_variable_names(tf.all_variables())
        self._saver = tf.train.Saver(var_list=var_map,
                                     max_to_keep=max_tf_checkpoints_to_keep)

    def _create_actor_network(self, name):
        actor_network = self.actor_network(self.num_actions, name=name)
        return actor_network

    def _build_networks(self):
        self.actor = self._create_actor_network('actor')
        self.action_logits = self.actor_out_fn(self.actor(self.states_ph),
                                               axis=1)

    def _build_train_op(self):
        log_probs = tf.log(self.action_logits)
        labels = tf.one_hot(self.actions_ph, self.num_actions, axis=1)
        neg_log_probs = -tf.reduce_sum(log_probs * labels, axis=1)
        self._loss = tf.reduce_mean(neg_log_probs * self.rewards_ph)
        # self._loss = trfl.discrete_policy_gradient(
        #     self.action_logits, self.actions_ph, self.rewards_ph)
        return self.actor_optimizer.minimize(self._loss)

    def begin_episode(self, observation):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminal[:]
        self._reset_state()
        self._record_observation(observation)
        self.action = self._select_action()

        return self.action

    def step(self, reward, observation):
        self._last_observation = self._observation
        self._record_observation(observation)
        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward,
                                   False)
        self.action = self._select_action()
        return self.action

    def end_episode(self, reward):
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward,
                                   True)
            self._train_step()

    def _select_action(self):
        probs = self._sess.run(self.action_logits,
                               feed_dict={self.states_ph: self.state})
        action = np.random.choice(range(probs.shape[1]), p=probs.ravel())
        self.actions.append(action)
        return action

    def _train_step(self):
        self._sess.run(self._train_op,
                       feed_dict={
                           self.actions_ph:
                           self.actions,
                           self.rewards_ph:
                           discount_reward(self.rewards, gamma=self.gamma),
                           self.states_ph:
                           np.vstack(self.states)
                       })

    def _record_observation(self, observation):
        self._observation = np.reshape(observation, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[0, ...] = self._observation
        self.states.append(self.state)

    def _store_transition(self, _last_observation, action, reward,
                          is_terminal):
        self.rewards.append(reward)
        self.is_terminal.append(is_terminal)

    def _reset_state(self):
        self.state.fill(0)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        if not tf.gfile.Exists(checkpoint_dir):
            return None
        self._saver.save(self._sess,
                         os.path.join(checkpoint_dir, 'tf_ckpt'),
                         global_step=iteration_number)
        bundle_dictionary = {}
        bundle_dictionary['state'] = self.state
        bundle_dictionary['training_steps'] = self.training_steps
        return bundle_dictionary

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
        if bundle_dictionary is not None:
            for key in self.__dict__:
                if key in bundle_dictionary:
                    self.__dict__[key] = bundle_dictionary[key]
        elif not self.allow_partial_reload:
            return False
        else:
            tf.logging.warning("Unable to reload the agent's parameters!")
        # Restore the agent's TensorFlow graph.
        self._saver.restore(
            self._sess,
            os.path.join(checkpoint_dir,
                         'tf_ckpt-{}'.format(iteration_number)))
        return True