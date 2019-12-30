from dopamine.discrete_domains import atari_lib
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections

import gin.tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
import gym_minigrid

gin.constant('minigrid_lib.MINIGRID_OBSERVATION_SHAPE', (56, 56, 3))
gin.constant('minigrid_lib.MINIGRID_OBSERVATION_DTYPE', tf.uint8)

RainbowNetworkType = collections.namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])


@gin.configurable
def create_minigrid_environment(game_name=None):
    assert game_name is not None
    env = gym.make(game_name)
    env = MinigridPreprocessing(env)
    return env


@gin.configurable
class MinigridPreprocessing(object):
    """A Wrapper class around Gym environments."""
    def __init__(self, environment):
        environment = gym_minigrid.wrappers.RGBImgPartialObsWrapper(
            environment)
        environment = gym_minigrid.wrappers.ImgObsWrapper(environment)
        self.environment = environment
        self.game_over = False
        self.tile_size = 8

    @property
    def observation_space(self):
        return self.environment.observation

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def reset(self):
        return self.environment.reset()

    def step(self, action):
        observation, reward, game_over, info = self.environment.step(action)
        self.game_over = game_over
        return observation, reward, game_over, info


@gin.configurable
class BasicDiscreteDomainNetwork(tf.keras.layers.Layer):
    def __init__(self,
                 num_actions,
                 num_atoms=None,
                 name=None,
                 activation_fn=tf.keras.activations.relu):
        super(BasicDiscreteDomainNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        # Defining layers.
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1 = tf.keras.layers.Conv2D(16, [2, 2],
                                            strides=1,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(32, [2, 2],
                                            strides=1,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [2, 2],
                                            strides=1,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.dense1 = tf.keras.layers.Dense(64,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(64,
                                            activation=activation_fn,
                                            name='fully_connected')
        if num_atoms is None:
            self.last_layer = tf.keras.layers.Dense(num_actions,
                                                    name='fully_connected')
        else:
            self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms,
                                                    name='fully_connected')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)

        if tf.rank(x) != 4:
            x_shape = tf.shape(x)
            x = tf.reshape(
                x, [-1, x_shape[1], x_shape[2],
                    tf.reduce_prod(x_shape[3:])])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.batchnorm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.last_layer(x)
        return x


@gin.configurable
class MinigridDQNNetwork(tf.keras.Model):
    def __init__(self, num_actions, name=None):
        """Builds the deep network used to compute the agent's Q-values.

    It rescales the input features so they lie in range [-1, 1].

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
        super(MinigridDQNNetwork, self).__init__(name=name)
        self.net = BasicDiscreteDomainNetwork(num_actions)

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input."""
        x = self.net(state)
        return atari_lib.DQNNetworkType(x)


@gin.configurable
class MinigridPGNetwork(tf.keras.Model):
    def __init__(self, num_actions, name=None):
        super(MinigridPGNetwork, self).__init__(name=name)
        self.net = BasicDiscreteDomainNetwork(num_actions)

    def call(self, state):
        x = self.net(state)
        return x


@gin.configurable
class MinigridRainbowNetwork(tf.keras.Model):
    """The convolutional network used to compute agent's return distributions."""
    def __init__(self, num_actions, num_atoms, support, name=None):
        """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      name: str, used to crete scope for network parameters.
    """
        super(MinigridRainbowNetwork, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.support = support
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        # Defining layers.
        self.conv1 = tf.keras.layers.Conv2D(
            16, [2, 2],
            strides=1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self.kernel_initializer,
            name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(
            32, [2, 2],
            strides=1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self.kernel_initializer,
            name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [2, 2],
            strides=1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self.kernel_initializer,
            name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            64,
            activation=activation_fn,
            kernel_initializer=self.kernel_initializer,
            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(
            num_actions * num_atoms,
            kernel_initializer=self.kernel_initializer,
            name='fully_connected')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x_shape = tf.shape(x)
        x = tf.reshape(
            x, [-1, x_shape[1], x_shape[2],
                tf.reduce_prod(x_shape[3:])])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(logits)
        q_values = tf.reduce_sum(self.support * probabilities, axis=2)
        std_values = tf.reduce_sum(
            (probabilities * self.num_atoms - tf.expand_dims(q_values, -1)**2),
            -1)

        def value_fn(r, b=0.1, c=0.001):
            return b * tf.log(c * r + 1)

        def dist_value_fn(mean, std):
            return value_fn(mean) - 0.1 / (2 * mean**2) * std

        # q_values = dist_value_fn(q_values, std_values)

        return RainbowNetworkType(q_values, logits, probabilities)
