from dopamine.discrete_domains import atari_lib
import gym
import numpy as np
import tensorflow.compat.v1 as tf

import gin.tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
import gym_minigrid

gin.constant('minigrid_lib.MINIGRID_OBSERVATION_SHAPE', (56, 56, 3))
gin.constant('minigrid_lib.MINIGRID_OBSERVATION_DTYPE', tf.uint8)


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
    """The fully connected network used to compute the agent's Q-values.

    This sub network used within various other models. Since it is an inner
    block, we define it as a layer. These sub networks normalize their inputs to
    lie in range [-1, 1], using min_/max_vals. It supports both DQN- and
    Rainbow- style networks.
    Attributes:
      min_vals: float, minimum attainable values (must be same shape as
        `state`).
      max_vals: float, maximum attainable values (must be same shape as
        `state`).
      num_actions: int, number of actions.
      num_atoms: int or None, if None will construct a DQN-style network,
        otherwise will construct a Rainbow-style network.
      name: str, used to create scope for network parameters.
      activation_fn: function, passed to the layer constructors.
  """
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
        self.dense1 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        if num_atoms is None:
            self.last_layer = tf.keras.layers.Dense(num_actions,
                                                    name='fully_connected')
        else:
            self.last_layer = tf.keras.layers.Dense(num_actions * num_atoms,
                                                    name='fully_connected')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input."""
        x = tf.cast(state, tf.float32)
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
