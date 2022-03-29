import os
import cv2
import random
import numpy as np
from tqdm import tqdm


def noise(X):
    return X.copy() + np.random.standard_normal(X.shape)


class DataManager:
    def __init__(self, size=-1):
        self.X = None
        self.set_size = None
        self.load_data(size)

    def load_data(self, size):         
        images = os.listdir("./celeba/")
        random.shuffle(images)
        if size != -1:
            images = images[:size]

        data = []
        for i in tqdm(images):
            img = cv2.cvtColor(cv2.imread("./celeba/"+i), cv2.COLOR_BGR2GRAY)
            data.append(cv2.resize(img, (32, 32)))

        self.X = np.array(data)
        self.set_size = len(images)

    def get_batch(self, batch_size, index=0):
        start = index*batch_size
        end = start+batch_size
        size = end-start

        x = np.append(self.X[start:end], noise(self.X[start:end]), axis=0)
        y = np.append(np.ones(size), np.zeros(size), axis=0)

        indices = np.arange(size*2)
        np.random.shuffle(indices)

        return y[indices], x[indices]

    def shuffle(self):
        indices = np.arange(self.set_size)
        np.random.shuffle(indices)
        self.X = self.X[indices]
