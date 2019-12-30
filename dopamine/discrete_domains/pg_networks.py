import tensorflow as tf
import gin


@gin.configurable
class ActorNetwork(tf.keras.Model):
    def __init__(self,
                 num_actions,
                 name=None,
                 activation_fn=tf.keras.activations.relu):
        super(ActorNetwork, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8],
                                            strides=4,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4],
                                            strides=2,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3],
                                            strides=1,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(num_actions,
                                            name='fully_connected')

    def call(self, state):
        
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


@gin.configurable
class CriticNetwork(tf.keras.Model):
    def __init__(self, name=None, activation_fn=tf.keras.activations.relu):
        super(CriticNetwork, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8],
                                            strides=4,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4],
                                            strides=2,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3],
                                            strides=1,
                                            padding='same',
                                            activation=activation_fn,
                                            name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state):
        
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x


@gin.configurable
class CartpoleActorNetwork(tf.keras.Model):
    def __init__(self,
                 num_actions,
                 name=None,
                 activation_fn=tf.keras.activations.relu):
        super(CartpoleActorNetwork, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.last_layer = tf.keras.layers.Dense(num_actions,
                                                name='fully_connected')

    def call(self, state):
        
        x = tf.cast(state, tf.float32)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.last_layer(x)
        return x


@gin.configurable
class CartpoleCriticNetwork(tf.keras.Model):
    def __init__(self, name=None, activation_fn=tf.keras.activations.relu):
        super(CartpoleCriticNetwork, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(512,
                                            activation=activation_fn,
                                            name='fully_connected')
        self.last_layer = tf.keras.layers.Dense(1, name='fully_connected')

    def call(self, state):
        
        x = tf.cast(state, tf.float32)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.last_layer(x)
        return x