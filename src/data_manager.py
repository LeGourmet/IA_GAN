import os
import cv2
import random
import numpy as np
from tqdm import tqdm


class DataManager:
    def __init__(self, size=-1):
        self.X = None
        self.set_size = None
        self.load_data(size)

    def load_data(self, size):
        images = os.listdir("./big_celeba/")
        random.shuffle(images)
        if size != -1:
            images = images[:size]

        data = []
        for i in tqdm(images):
            img = cv2.cvtColor(cv2.imread("./big_celeba/" + i), cv2.COLOR_BGR2GRAY)
            data.append(cv2.resize(img, (32, 32)))

        self.X = np.array(data)
        self.X = np.reshape(self.X, (self.X.shape[0], 1024))
        self.set_size = len(images)

    def get_batch(self, batch_size, index=0):
        start = index * batch_size
        end = start + batch_size

        return self.X[start:end]

    def shuffle(self):
        indices = np.arange(self.set_size)
        np.random.shuffle(indices)
        self.X = self.X[indices]
