"""
Implementation of:
https://arxiv.org/pdf/1512.09300.pdf
"""


import tensorflow as tf
from time import time
import numpy as np
from itertools import product

from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer, Layer, MaxPool2D, AveragePooling2D, BatchNormalization, Dropout, ReLU, LeakyReLU
from tensorflow.keras import Model
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, 'train samples')
print(x_test.shape[0], 'test samples')


x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]



train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class VAE(Model):

    def __init__(self, input_shape, latent_dim):
    
        super().__init__()
    
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(filters=16, kernel_size=3, strides=2, activation='relu'), 
                BatchNormalization(),
                Conv2D(filters=32, kernel_size=3, strides=2, activation='relu'), 
                BatchNormalization(),
                Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'), 
                BatchNormalization(),
                Conv2D(filters=128, kernel_size=3, strides=1, activation='relu'), 
                Flatten(),
                Dense(latent_dim + latent_dim),
            ],
            name = 'encoder'
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=2**2*128, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(-1, 1, 2**2*128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16,
                    kernel_size=4,
                    strides=1,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1,
                    kernel_size=4,
                    strides=2,
                    activation=None),
               
            ],
            name = 'decoder'
        )

    def call(self, x, training=True):
        mu, logvar = self.encode(x, training=training)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, apply_sigmoid=True)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


    def encode(self, x, training=True):
        mu, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu

    def decode(self, z, apply_sigmoid=False):
        
        out = self.decoder(z)
        if apply_sigmoid:
            out = tf.sigmoid(out) # not logits
        return out 

    def get_latent(self, x):
        mu, logvar = self.encode(x, training=False)
        return self.reparameterize(mu, logvar)

    def test(self, x):
        n = 5
        samples = np.random.choice(len(x), n**2, replace=False)
        #samples.sort()
        x = np.array(x)
        x = x[samples,:]
        #print(x[0])
        #print(x.shape)
        preds = self.call(x, training=False)
        preds = np.array(preds)
        #print(preds[0])
        #print(preds)
        imd = 28
        canvas_orig = np.empty((imd*n , 2*imd * n+1, 1))
        for i in range(n):
            batch_x = x[i*n:i*n+n]
            g = preds[i*n:i*n+n]

            for j in range(n):
                canvas_orig[i * imd:(i + 1) * imd, j * imd:(j + 1) * imd] = \
                    batch_x[j].reshape([imd, imd, 1])
                canvas_orig[i * imd :(i + 1) * imd, j * imd + n*imd+1:(j + 1) * imd + n*imd+1] = \
                    g[j].reshape([imd, imd, 1])
        canvas_orig[:, n*imd:n*imd+1] = 1
        print(canvas_orig.shape)

        print("Original Images")
        plt.figure(figsize=(n*2+1, n))
        plt.imshow(canvas_orig[:,:,0], origin="upper", cmap='gray')
        plt.draw()
        plt.show()


#@tf.function
def compute_loss(vae_generator, discriminator, x, beta):
    mean, logvar = vae_generator.encode(x)
    z = vae_generator.reparameterize(mean, logvar)
    y = vae_generator.decoder(z)

    #print(y.shape)
    #print(x.shape)

    real_output = discriminator(x, training=True)
    fake_output = discriminator(y, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)



    rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x), axis=[1,2,3])
    rec_loss = tf.reduce_mean(rec_loss)



    kl_loss = -0.5*tf.math.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar)), axis=1)
    #kl_loss *= 0
    # https://openreview.net/forum?id=Sy2fzU9gl
    #kl_loss *= beta
    kl_loss = tf.reduce_mean(kl_loss)

    print('kl', kl_loss)
    print('recl', rec_loss)
    print('gen_l', gen_loss)
    print('discl', disc_loss)

    disc_loss *= 0.5


    en_loss = rec_loss+kl_loss
    de_loss = 0.4*rec_loss+gen_loss

    return en_loss, de_loss, disc_loss


def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)



if __name__=="__main__":
    input_shape = (28, 28, 1)
    disc_layers = [
        InputLayer(input_shape=input_shape),
        Conv2D(32, 3, strides=1, padding='same'),
        LeakyReLU(),
        Conv2D(64, 4, strides=2, padding='same'),
        LeakyReLU(),
        Conv2D(64, 4, strides=2, padding='same'),
        LeakyReLU(),
        Flatten(),
        Dense(1)
    ]

    discriminator = tf.keras.Sequential(disc_layers, name="discriminator")
    discriminator.summary()

    generator = VAE(input_shape=input_shape, latent_dim=256)
    generator.summary()

    encoder_optimizer = tf.keras.optimizers.Adam(5e-4)
    decoder_optimizer = tf.keras.optimizers.Adam(5e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(5e-4)
    # for test_images, labels in test_ds:
    #     #generator.test(test_images)
    #     break

    
    beta = tf.Variable(1.)
    #print(dir(train_ds))
    for e in range(100):
        st = time()
        for i, (images, labels) in enumerate(train_ds):
            print(i)
            if not (i+1)%234:
                print('hello')
                for test_images, labels in test_ds:
                    generator.test(test_images)
                    break
            images = tf.dtypes.cast(images, tf.float32)

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3: 
                en_loss, de_loss, disc_loss = compute_loss(generator, discriminator, images, beta)

            
            gradients_of_encoder = tape1.gradient(en_loss, generator.encoder.trainable_variables)
            gradients_of_decoder = tape2.gradient(de_loss, generator.decoder.trainable_variables)
            gradients_of_discriminator = tape3.gradient(disc_loss, discriminator.trainable_variables)

            encoder_optimizer.apply_gradients(zip(gradients_of_encoder, generator.encoder.trainable_variables))
            decoder_optimizer.apply_gradients(zip(gradients_of_decoder, generator.decoder.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        print(time()-st)

        for test_images, labels in test_ds:
            generator.test(test_images)
            break
