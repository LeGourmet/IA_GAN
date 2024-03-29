import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageGenerator.data_manager import DataManager
from imageGenerator.model import *
from sample import sample
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

flags.DEFINE_integer("epochs", 100, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.005, "learning rate")
FLAGS = flags.FLAGS


def train(gan):
    manager = DataManager(1000)
    lossTab = []
    loss = 0

    HBS = FLAGS.batch_size
    FBS = 2*FLAGS.batch_size
    nbBatches = manager.size // FLAGS.batch_size
    resRealImage = np.ones(HBS)
    resFakeImage = np.zeros(HBS)
    resGan = np.ones(FBS)

    for epoch in range(1,FLAGS.epochs+1):
        print('Epoch', epoch, '/', FLAGS.epochs)
        manager.shuffle()

        for i in tqdm(range(nbBatches)):
            zH = np.random.randint(manager.size, size=HBS)
            zF = np.random.randint(manager.size, size=FBS)

            gan.disc.model.trainable = True
            lossReal = gan.disc.model.train_on_batch(manager.get_batch(HBS, i), resRealImage)
            lossFake = gan.disc.model.train_on_batch(gan.gen.model.predict(zH), resFakeImage)
            loss = (1-lossReal + lossFake)/2    # like in function discriminator_loss
            gan.disc.model.trainable = False
            gan.model.train_on_batch(zF, resGan)

        lossTab.append(loss)
        print("Epoch {} - loss: {}".format(epoch, loss))

    print("Finished training.")

    # gan.disc.model.save("./trained_model/disc_model.h5")
    # gan.disc.model.save("./trained_model/gen_model.h5")
    sample(gan)

    plt.plot(lossTab)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def load(path, model):
    if os.path.isfile(path):
        print("Loading model from", path)
        model.load_weights(path)
    else:
        print(path, "not found")


def load_discriminator():
    model_disc = Discriminator()
    load("./trained_model/disc_model.h5", model_disc.model)
    model_disc.model.compile(loss=BinaryCrossentropy(), optimizer=Adam(FLAGS.learning_rate))
    model_disc.model.summary()
    return model_disc


def load_generator():
    model_gen = Generator()
    load("./trained_model/gen_model.h5", model_gen.model)
    model_gen.model.summary()
    return model_gen


def main(argv):
    disc = load_discriminator()
    gen = load_generator()
    gan = GAN(disc, gen)
    gan.model.compile(loss=BinaryCrossentropy(), optimizer=Adam(FLAGS.learning_rate))
    gan.model.summary()
    train(gan)


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
