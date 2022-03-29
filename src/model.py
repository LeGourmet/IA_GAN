import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Reshape


class Discriminator:
    def __init__(self):
        self.model = None
        self.make_discriminator()
        
    def make_discriminator(self):
        '''
        model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        puis fatten
        '''

        self.model = tf.keras.Sequential()
        self.model.add(Input(shape=(32, 32), name="discriminator_input"))
        self.model.add(Dense(25, activation='elu'))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid', name="discriminator_output"))


class Generator:
    def __init__(self):
        self.model = None
        self.make_generator()
    
    def make_generator(self):
        self.model = tf.keras.Sequential()
        self.model.add(Input(shape=1, name="generator_input"))
        self.model.add(Dense(15, activation='elu'))
        self.model.add(Dense(1024, activation='elu'))
        self.model.add(Reshape((32, 32), name="generator_output"))


class GAN:
    def __init__(self, dis, gen):
        self.model = None
        self.disc = None
        self.gen = None
        self.make_GAN(dis, gen)
        
    def make_GAN(self, disc, gen):
        self.disc = disc
        self.gen = gen

        self.model = tf.keras.Sequential()
        self.model.add(self.gen.model)
        self.disc.model.trainable = False
        self.model.add(self.disc.model)
