"""
TF2.0 implementation of SqueezeNet (https://arxiv.org/pdf/1602.07360.pdf)
Get weights from: http://cs231n.stanford.edu/squeezenet_tf.zip
"""

import tensorflow as tf 
import numpy as np 

from tensorflow.keras.layers import Layer, Dense, Conv2D, InputLayer, MaxPool2D, AveragePooling2D


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class FireLayer(tf.keras.Model):
    def __init__(self, squeeze_filters, conv11_filters, conv33_filters, **kwargs):
        super(FireLayer, self).__init__(**kwargs)

        self.squeeze = Conv2D(filters=squeeze_filters, kernel_size=1, strides=1, activation='relu')
        self.conv11 = Conv2D(filters=conv11_filters, kernel_size=1, strides=1, activation='relu')
        self.conv33 = Conv2D(filters=conv33_filters, kernel_size=3, strides=1, activation='relu', padding='SAME')
        

    def call(self, x):
        squee = self.squeeze(x)
        conv11_out = self.conv11(squee)
        conv33_out = self.conv33(squee)
        fin_out = tf.concat([conv11_out, conv33_out], 3)
        return fin_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class SqueezeNet(tf.keras.Model):

    def __init__(self, layers_return=None):
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
                Conv2D(filters=1000, kernel_size=1, strides=1, activation='relu'), #12
                AveragePooling2D(pool_size=13, strides=13),
            ]
        )  
        self.layers_return = layers_return


    def get_layers(self,x, layers_return=None):
        l_out = []
        next_out = x
        for i, layer in enumerate(self.model.layers):
            #print(layer)
            next_out = layer(next_out)
            if layers_return:
                if i in layers_return:
                    l_out.append(next_out)
            else:
                l_out.append(next_out)

        return l_out

    def call(self, x):
        if not self.layers_return:
            return self.model(x)
        else:
            return self.get_layers(x, self.layers_return)



    def preprocess_image(self, img):
        """
        Preprocess an image for SqueezeNet.
        Subtracts the pixel mean and divides by the standard deviation.
        """
        return (img.astype(np.float32)/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


    def deprocess_image(self, img, rescale=False):
        """Undo preprocessing on an image and convert back to uint8."""
        img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
        img = np.array(img)
        if rescale:
            vmin, vmax = img.min(), img.max()
            img = (img - vmin) / (vmax - vmin)
        return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


    def clip_image(self, img):
        return tf.clip_by_value(img, -1.5, 1.5)