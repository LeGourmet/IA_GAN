import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from absl import app
from absl import flags
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_manager import DataManager
from model import *
from sample import sample
import random as rd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

flags.DEFINE_integer("epochs", 5, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
flags.DEFINE_boolean("keep_training", True, "continue training same weights")
flags.DEFINE_boolean("keep_best", True, "only save model if it got the best loss")
FLAGS = flags.FLAGS

best_loss = np.inf
model_path = "./trained_model/"

def train(gan):
    manager = DataManager(10000)
    loss = []
    batches = manager.set_size // FLAGS.batch_size

    for epoch in range(FLAGS.epochs):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()

        for i in tqdm(range(batches)):
            # z = stats.laplace.pdf(np.linspace(-5, 5, batches))
            # z = np.random.standard_normal(batches)
            z = stats.laplace.cdf([rd.random() for _ in range(batches)])
            #z = tf.random.normal(shape=(batches, 1))

            gan.disc.model.trainable = True
            X, Y = manager.get_batch(FLAGS.batch_size, i)
            gan.disc.model.train_on_batch(X, Y)
            gan.disc.model.train_on_batch(gan.gen.model.predict(z), np.zeros(batches))
            gan.disc.model.trainable = False

            l = gan.model.train_on_batch(z, np.ones(batches))
        loss.append(l)

        print("Epoch {} - loss: {}".format(epoch, loss[epoch]))
    print("Finished training.")

    sample(gan)

    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def save_model(name, model, loss):
    global best_loss
    if not FLAGS.keep_best:
        model.save(model_path+name)
    elif loss < best_loss:
        best_loss = loss
        model.save(model_path+name)


def load_discriminator():
    model_disc = Discriminator()

    if os.path.isfile(model_path+"disc_model.h5"):
        print("Loading model from", model_path+"disc_model.h5")
        model_disc.model.load_weights(model_path)

    model_disc.model.compile(loss=BinaryCrossentropy(), optimizer=Adam(FLAGS.learning_rate))
    model_disc.model.summary()
    return model_disc


def load_generator():
    model_gen = Generator()

    if os.path.isfile(model_path+"gen_model.h5"):
        print("Loading model from", model_path+"gen_model.h5")
        model_gen.model.load_weights(model_path)

    model_gen.model.summary()
    return model_gen


def main(argv):
    disc = load_discriminator()

    gen = load_generator()

    gan = GAN(disc, gen)
    gan.model.compile(loss=BinaryCrossentropy(), optimizer=Adam(FLAGS.learning_rate))

    train(gan)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
