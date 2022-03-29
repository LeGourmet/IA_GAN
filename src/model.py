import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten, Reshape


class Discriminator:
    def __init__(self):
        self.model = None
        self.make_discriminator()
        
    def make_discriminator(self):
        self.model = tf.keras.Sequential()
        self.model.add(Input(shape=(32, 32), name="discriminator_input"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu', input_dim=1))
        self.model.add(Dense(128, activation='relu', input_dim=1))
        self.model.add(Dense(32, activation='relu', input_dim=1))
        self.model.add(Dense(1, activation='sigmoid', name="discriminator_output"))
    
    
class Generator:
    def __init__(self):
        self.model = None
        self.make_generator()
    
    def make_generator(self):
        self.model = tf.keras.Sequential()
        self.model.add(Input(shape=(1), name="generator_input"))
        self.model.add(Dense(32, activation='relu', input_dim=1))
        self.model.add(Dense(128, activation='relu', input_dim=1))
        self.model.add(Dense(512, activation='relu', input_dim=1))
        self.model.add(Dense(1536, activation='relu', input_dim=1))
        self.model.add(Dense(1024, activation='relu', input_dim=1))
        self.model.add(Reshape((32, 32), name="generator_output"))

    def generator_loss(self, fake_output):
        return tf.reduce_mean(-tf.math.log(fake_output))


class GAN:
    def __init__(self, des, gen):
        self.model = None
        self.des = None
        self.gen = None
        self.make_GAN(des, gen)
        
    def make_GAN(self, des, gen):
        self.model = tf.keras.Sequential()
        self.gen = gen
        self.model.add(gen.model)
        self.des = des
        self.model.add(des.model)
