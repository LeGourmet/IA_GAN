from absl import app
from absl import flags

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from data_manager import DataManager
from model import *
from sample import sample

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

from tensorflow.keras.optimizers import Adam

flags.DEFINE_integer("epochs", 200, "number of epochs")
flags.DEFINE_integer("batch_size", 5, "batch size")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_boolean("keep_training", True, "continue training same weights")
flags.DEFINE_boolean("keep_best", True, "only save model if it got the best loss")
FLAGS = flags.FLAGS

best_loss = np.inf
model_path = "./trained_model/"


def train(gan):
    manager = DataManager()
    loss = []

    for epoch in range(FLAGS.epochs):
        print('Epoch', epoch, '/', FLAGS.epochs)

        gan.des.model.trainable = True
        manager.shuffle()
        Y, X = manager.get_batch(FLAGS.batch_size)
        gan.des.model.train_on_batch(X, Y)
        gan.des.model.trainable = False

        gan.gen.model.trainable = True
        Y, X = (np.ones(FLAGS.batch_size), stats.laplace.pdf(np.linspace(-32, 32, FLAGS.batch_size)))
        loss.append(gan.model.train_on_batch(X, Y))

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

    model_disc.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(FLAGS.learning_rate))
    model_disc.model.summary()
    return model_disc


def load_generator():
    model_gen = Generator()

    if os.path.isfile(model_path+"gen_model.h5"):
        print("Loading model from", model_path+"gen_model.h5")
        model_gen.model.load_weights(model_path)

    model_gen.model.summary()
    return model_gen


def load_Gan(dis, gen):
    model_gan = GAN(dis, gen)
    model_gan.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(FLAGS.learning_rate))
    model_gan.model.summary()
    return model_gan


def main(argv):
    dis = load_discriminator()
    gen = load_generator()
    gan = load_Gan(dis, gen)
    train(gan)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
