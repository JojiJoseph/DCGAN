import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tqdm import tqdm


class Generator(tfk.Model):
    def __init__(self):
        super().__init__()
        # self.l1 = tfkl.Dense(1024, activation="leaky_relu")
        # self.bn1 = tfkl.BatchNormalization()
        # self.l2 = tfkl.Dense(784, activation="sigmoid")
        self.l1 = tfkl.Dense(4*4*1024, activation="leaky_relu")
        self.c1 = tfkl.Conv2DTranspose(64,5,strides=2, activation="leaky_relu")
        self.b1 = tfkl.BatchNormalization()
        self.c2 = tfkl.Conv2DTranspose(64,5,strides=2, activation="leaky_relu")
        self.b2 = tfkl.BatchNormalization()
        self.c3 = tfkl.Conv2DTranspose(1,4,strides=1, activation="sigmoid")
    def call(self, x):
        y = self.l1(x)
        y = tf.reshape(y, (-1,4,4,1024))
        y = self.c1(y)
        y = self.b1(y)
        y = self.c2(y)
        y = self.b2(y)
        y = self.c3(y)
        # print(y.shape)
        return y

class Disc(tfk.Model):
    def __init__(self):
        super().__init__()
        # self.l1 = tfkl.Dense(512, activation="leaky_relu")
        # self.l2 = tfkl.Dense(1, activation="sigmoid")
        self.c1 = tfkl.Conv2D(64, 3, 2, activation="leaky_relu")
        self.b1 = tfkl.BatchNormalization()
        self.c2 = tfkl.Conv2D(64, 3, 2, activation="leaky_relu")
        self.b2 = tfkl.BatchNormalization()
        self.c3 = tfkl.Conv2D(64, 3, 2, activation="leaky_relu")
        # self.b3 = tfkl.BatchNormalization()
        # self.c4 = tfkl.Conv2D(64, 3, 2, activation="leaky_relu")
        # self.l1 = tfkl.Dense(512, activation="leaky_relu")
        # self.l2 = tfkl.Dense(1, activation="sigmoid")
        self.l1 = tfkl.Dense(1, activation="sigmoid")
    def call(self, x):
        # x = tf.reshape(x,(-1,28*28))
        y = self.c1(x)
        y = self.b1(y)
        y = self.c2(y)
        y = self.b2(y)
        y = self.c3(y)
        # y = self.b3(y)
        y = tf.reshape(y, (x.shape[0], -1))
        y = self.l1(y)
        # y = self.l2(y)
        # print(y.shape)
        return y

if __name__ == "__main__":
    gen = Generator()
    gen(np.zeros((1,100)))