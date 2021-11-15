import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tqdm import tqdm


class Generator(tfk.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tfkl.Embedding(10, 20)
        self.seq = tfk.models.Sequential([
            tfkl.Conv2DTranspose(1024,4,use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            # 4x4x1024
            tfkl.Conv2DTranspose(512,4,strides=(2,2),padding='same',use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            # 8x8x512
            tfkl.Conv2DTranspose(256,4,strides=(2,2),padding='same',use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            # 16x16x256
            tfkl.Conv2DTranspose(128,4,strides=(2,2),padding='same',use_bias=False),
            # tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            # 32x32x128
            tfkl.Conv2D(1,5,use_bias=False),
            tfkl.Activation(tf.nn.tanh)
            # 28x28x1
        ])
    def call(self, x, category):
        cat = self.embedding(category)
        x = tf.concat([x,cat], axis=-1)
        y = tf.reshape(x, (-1,1,1,120))
        y = self.seq(y)
        return y

class Disc(tfk.Model):
    def __init__(self):
        super().__init__()

        self.embedding = tfkl.Embedding(10, 28*28)

        self.seq = tfk.models.Sequential([
            tfkl.Conv2D(32,5,activation="leaky_relu", use_bias=False),
            
            tfkl.Conv2D(64,4,strides=2, use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),

            tfkl.Conv2D(128,4,strides=2, use_bias=False),
            tfkl.LeakyReLU(alpha=0.2),

            tfkl.Conv2D(1,4, activation="sigmoid", use_bias=False),
            tfkl.Flatten()
        ])
    def call(self, x, category):
        cat = self.embedding(category)
        cat = tf.reshape(cat, (-1,28,28,1))
        x = tf.concat([x,cat],axis=-1)
        y = self.seq(x)
        return y
