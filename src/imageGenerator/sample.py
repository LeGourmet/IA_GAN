import os
import random as rd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import *
from absl import app

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def sample(gan):
    X = gan.gen.model.predict([rd.random()])
    print("It's real at :", gan.disc.model.predict(X)[0][0])
    plt.imshow(np.reshape(X, (32, 32)), cmap='gray', vmin=0, vmax=255)
    plt.show()


def load(path, model):
    if os.path.isfile(path):
        print("Loading model from", path)
        model.load_weights(path)
    else:
        print(path, "not found")


def load_model():
    model_gen = Generator()
    load("./trained_model/gen_model.h5", model_gen.model)
    model_gen.model.summary()

    model_disc = Discriminator()
    load("./trained_model/disc_model.h5", model_disc.model)
    model_disc.model.summary()

    gan = GAN(model_disc, model_gen)
    gan.model.summary()
    return gan


def main(argv):
    sample(load_model())


if __name__ == '__main__':
    app.run(main)
