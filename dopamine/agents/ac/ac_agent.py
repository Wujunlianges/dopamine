import tensorflow as tf
import numpy as np
from dopamine.agents.ac.ac_agent import ACAgent
from pg_networks import ActorNetwork, CriticNetwork
from dopamine.discrete_domains import run_experiment, atari_lib
import gin


@gin.configurable
class ACAgent(PGAgent):
    def __init__(
            self,
            sess,
            num_actions,
            actor_dist_fn=tf.distributions.Categorical,
            observation_shape=atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
            observation_dtype=atari_lib.NATURE_DQN_DTYPE,
            actor_network=ActorNetwork,
            critic_network=CriticNetwork,
            gamma=0.99,
            update_horizon=1,
            actor_out_fn=tf.nn.softmax,
            tf_device='/cpu:*',
            eval_mode=False,
            use_staging=False,
            max_tf_checkpoints_to_keep=4,
            actor_optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025,
                                                      decay=0.95,
                                                      momentum=0.0,
                                                      epsilon=0.00001,
                                                      centered=True),
            critic_optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025,
                                                       decay=0.95,
                                                       momentum=0.0,
                                                       epsilon=0.00001,
                                                       centered=True),
            summary_writer=None,
            summary_writing_frequency=500,
            allow_partial_reload=False):
        self.critic_network = critic_network
        self.critic_optimizer = critic_optimizer
        PGAgent.__init__(self,
                         sess=sess,
                         num_actions=num_actions,
                         observation_shape=observation_shape,
                         observation_dtype=observation_dtype,
                         gamma=gamma,
                         update_horizon=update_horizon,
                         actor_network=actor_network,
                         tf_device=tf_device,
                         use_staging=use_staging,
                         max_tf_checkpoints_to_keep=max_tf_checkpoints_to_keep,
                         actor_optimizer=actor_optimizer,
                         summary_writer=summary_writer,
                         summary_writing_frequency=summary_writing_frequency,
                         allow_partial_reload=allow_partial_reload)

    def _create_critic_network(self, name):
        critic_network = self.critic_network(name=name)
        return critic_network

    def _build_networks(self):
        self.actor = self._create_actor_network('actor')
        self.critic = self._create_critic_network('critic')
        self.action_logits = self.actor_out_fn(self.actor(self.states_ph),
                                               axis=1)

    def _build_train_op(self):
        expected_v_values = tf.squeeze(self.critic(self.states_ph))
        expected_next_v_values = tf.roll(expected_v_values, shift=-1, axis=0)
        real_v_values = self.rewards_ph + (
            1 - self.is_terminal_ph) * self.gamma * expected_next_v_values
        td_error = real_v_values - expected_v_values

        log_probs = tf.log(self.action_logits)
        labels = tf.one_hot(self.actions_ph, self.num_actions, axis=1)
        neg_log_probs = -tf.reduce_sum(log_probs * labels, axis=1)
        self._actor_loss = tf.reduce_mean(neg_log_probs * td_error)
        self._critic_loss = tf.nn.l2_loss(expected_v_values - real_v_values)
        return [
            self.actor_optimizer.minimize(self._actor_loss),
            self.critic_optimizer.minimize(self._critic_loss)
        ]

    def _train_step(self):
        self._sess.run(self._train_op,
                       feed_dict={
                           self.actions_ph: self.actions,
                           self.rewards_ph: self.rewards,
                           self.states_ph: np.vstack(self.states),
                           self.is_terminal_ph: np.array(self.is_terminal)
                       })