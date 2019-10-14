"""
TF2.0 implementation of SqueezeNet (https://arxiv.org/pdf/1602.07360.pdf)
Get weights from: http://cs231n.stanford.edu/squeezenet_tf.zip
"""

import tensorflow as tf 
import numpy as np 

from tensorflow.keras.layers import Layer, Dense, Conv2D, InputLayer, MaxPool2D, AveragePooling2D



class FireLayer(Layer):
    def __init__(self, squeeze_filters, conv11_filters, conv33_filters, **kwargs):

        self.squeeze = Conv2D(filters=squeeze_filters, kernel_size=1, strides=1, activation='relu')
        self.conv11 = Conv2D(filters=conv11_filters, kernel_size=1, strides=1, activation='relu')
        self.conv33 = Conv2D(filters=conv33_filters, kernel_size=3, strides=1, activation='relu', padding='SAME')
        
        super(FireLayer, self).__init__(**kwargs)

    def call(self, x):
        squee = self.squeeze(x)
        conv11_out = self.conv11(squee)
        conv33_out = self.conv33(squee)
        fin_out = tf.concat([conv11_out, conv33_out], 3)
        return fin_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class SqueezeNet(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.model = tf.keras.Sequential(
            [
                InputLayer(input_shape=(227, 227, 3)),
                Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'), #0
                
                MaxPool2D(pool_size=3, strides=2), #1
                FireLayer(16, 64, 64), 
                FireLayer(16, 64, 64), #3
                MaxPool2D(pool_size=3, strides=2),
                FireLayer(32, 128, 128), #5                       
                FireLayer(32, 128, 128), #6
                MaxPool2D(pool_size=3, strides=2),
                FireLayer(48, 192, 192),
                FireLayer(48, 192, 192), #9
                FireLayer(64, 256, 256),
                FireLayer(64, 256, 256), #11
                # classifier
                Conv2D(filters=1000, kernel_size=1, strides=1, activation='relu'),
                AveragePooling2D(pool_size=13, strides=13),
            ]
        )

    def get_layers(self,x):
        l_out = []
        next_out = x
        for i, layer in enumerate(self.model.layers):
            #print(layer)
            next_out = layer(next_out)
            l_out.append(next_out)

        return l_out

    def call(self, x):
        return self.model(x)