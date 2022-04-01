import os
import sys
import random as rd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from absl import app
from absl import flags

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_integer("sample_size", 1, "samples to test")
flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
FLAGS = flags.FLAGS


def sample(gan):
    X = gan.gen.model.predict([rd.random()])
    print("I found :", gan.disc.model.predict(X))
    #X = 1/np.random.randint(1000, size=1024)
    plt.imshow(np.reshape(X, (32, 32)), cmap='gray')
    plt.show()


def load_model():
    model_path = os.path.abspath(FLAGS.model)
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model


def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    sample(load_model())


if __name__ == '__main__':
    app.run(main)
