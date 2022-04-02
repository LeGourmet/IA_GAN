import os
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from laplaceGenerator.model import *
from absl import app

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def sample(gan, nbsample=1000):
    bins = np.linspace(-10, 10, 41)
    lVal = np.random.uniform(-10, 10, size=nbsample)
    pred = (np.histogram(gan.gen.model.predict(lVal), bins)[0])/nbsample # normalize vector
    plt.bar(bins[:-1], pred, width=0.5)
    plt.plot(bins,stats.laplace.pdf(bins),'r')
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
