import tensorflow as tf
from tensorflow.keras.layers import Dense


class Discriminator:
    def __init__(self):
        self.model = None
        self.make_discriminator()
        
    def make_discriminator(self):
        self.model = tf.keras.Sequential()
        self.model.add(Dense(512, activation='elu', input_dim=1024, name="discriminator_input"))
        self.model.add(Dense(256, activation='elu'))
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(32, activation='elu'))
        self.model.add(Dense(1, activation='sigmoid', name="discriminator_output"))


class Generator:
    def __init__(self):
        self.model = None
        self.make_generator()
    
    def make_generator(self):
        self.model = tf.keras.Sequential()
        self.model.add(Dense(32, activation='elu', input_dim=1, name="generator_input"))
        self.model.add(Dense(64, activation='elu'))
        self.model.add(Dense(256, activation='elu'))
        self.model.add(Dense(512, activation='elu'))
        self.model.add(Dense(1024, activation='relu', name="generator_output"))


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
