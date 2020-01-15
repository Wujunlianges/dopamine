# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
from dopamine.replay_memory import prioritized_replay_buffer
import tensorflow.compat.v1 as tf

import gin.tf


@gin.configurable
class FourierAgent(dqn_agent.DQNAgent):
    """A compact implementation of a simplified Rainbow agent."""
    def __init__(self,
                 sess,
                 num_actions,
                 observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
                 observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
                 stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
                 network=atari_lib.RainbowNetwork,
                 num_atoms=51,
                 vmax=10.,
                 gamma=0.99,
                 update_horizon=1,
                 min_replay_history=20000,
                 update_period=4,
                 target_update_period=8000,
                 epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=250000,
                 replay_scheme='unifrom',
                 tf_device='/cpu:*',
                 use_staging=True,
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.00025,
                                                  epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500):
        # We need this because some tools convert round floats into ints.
        vmax = float(vmax)
        self._num_atoms = num_atoms
        self._support = tf.linspace(-vmax, vmax, num_atoms)
        self._replay_scheme = replay_scheme
        # TODO(b/110897128): Make agent optimizer attribute private.
        self.optimizer = optimizer

        dqn_agent.DQNAgent.__init__(
            self,
            sess=sess,
            num_actions=num_actions,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size,
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=self.optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)

    def _create_network(self, name):
        network = self.network(self.num_actions,
                               self._num_atoms,
                               self._support,
                               name=name)
        return network

    def _build_replay_buffer(self, use_staging):
        if self._replay_scheme not in ['uniform', 'prioritized']:
            raise ValueError('Invalid replay scheme: {}'.format(
                self._replay_scheme))
        # Both replay schemes use the same data structure, but the 'uniform' scheme
        # sets all priorities to the same value (which yields uniform sampling).
        return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _build_target_support(self):
        batch_size = self._replay.batch_size

        # size of rewards: batch_size x 1
        rewards = self._replay.rewards[:, None]

        # size of tiled_support: batch_size x num_atoms
        tiled_support = tf.tile(self._support, [batch_size])
        tiled_support = tf.reshape(tiled_support,
                                   [batch_size, self._num_atoms])

        # size of target_support: batch_size x num_atoms

        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals,
                                              tf.float32)
        # Incorporate terminal state to discount factor.
        # size of gamma_with_terminal: batch_size x 1
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
        gamma_with_terminal = gamma_with_terminal[:, None]

        target_support = rewards + gamma_with_terminal * tiled_support

        qt_argmax = tf.argmax(self._replay_net_outputs.q_values,
                              axis=1)[:, None]

        next_qt_argmax = tf.argmax(
            self._replay_next_target_net_outputs.q_values, axis=1)[:, None]

        batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
        # size of next_qt_argmax: batch_size x 2
        batch_indexed_next_qt_argmax = tf.concat(
            [batch_indices, next_qt_argmax], axis=1)

        batch_indexed_qt_argmax = tf.concat([batch_indices, qt_argmax], axis=1)

        next_logits = tf.gather_nd(self._replay_next_target_net_outputs.logits,
                                   batch_indexed_next_qt_argmax)

        logits = tf.gather_nd(self._replay_net_outputs.logits,
                              batch_indexed_qt_argmax)

        return target_support, next_logits, logits

    def _build_train_op(self):
        target_support, next_logits, logits = self._build_target_support()
        next_logits = tf.stop_gradient(next_logits)
        complex_logits = expo2complex(logits)
        complex_next_logits = expo2complex(next_logits)
        loss = (tf.math.real(complex_next_logits**2) * target_support +
                tf.math.real(complex_logits**2) * self._support)
        loss = tf.math.reduce_sum(tf.sqrt(loss), axis=-1)

        if self._replay_scheme == 'prioritized':
            # The original prioritized experience replay uses a linear exponent
            # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
            # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
            # a fixed exponent actually performs better, except on Pong.
            probs = self._replay.transition['sampling_probabilities']
            loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
            loss_weights /= tf.reduce_max(loss_weights)

            # Rainbow and prioritized replay are parametrized by an exponent alpha,
            # but in both cases it is set to 0.5 - for simplicity's sake we leave it
            # as is here, using the more direct tf.sqrt(). Taking the square root
            # "makes sense", as we are dealing with a squared loss.
            # Add a small nonzero value to the loss to avoid 0 priority items. While
            # technically this may be okay, setting all items to 0 priority will cause
            # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
            update_priorities_op = self._replay.tf_set_priority(
                self._replay.indices, tf.sqrt(loss + 1e-10))

            # Weight the loss by the inverse priorities.
            loss = loss_weights * loss
        else:
            update_priorities_op = tf.no_op()

        with tf.control_dependencies([update_priorities_op]):
            if self.summary_writer is not None:
                with tf.variable_scope('Losses'):
                    tf.summary.scalar('L2Loss', tf.reduce_mean(loss))
            # Schaul et al. reports a slightly different rule, where 1/N is also
            # exponentiated by beta. Not doing so seems more reasonable, and did not
            # impact performance in our experiments.
            return self.optimizer.minimize(tf.reduce_mean(loss)), loss

    def _store_transition(self,
                          last_observation,
                          action,
                          reward,
                          is_terminal,
                          priority=None):
        if priority is None:
            if self._replay_scheme == 'uniform':
                priority = 1.
            else:
                priority = self._replay.memory.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(last_observation, action, reward, is_terminal,
                             priority)


def expo2complex(expo):
    return tf.dtypes.complex(tf.math.sin(expo), tf.math.cos(expo))
