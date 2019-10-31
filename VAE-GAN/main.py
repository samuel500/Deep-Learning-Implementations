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

from tqdm import tqdm


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
                Conv2D(filters=16, kernel_size=4, strides=2, activation='relu'), 
                BatchNormalization(),
                Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'), 
                BatchNormalization(),
                Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'), 
                Flatten(),
                Dense(latent_dim + latent_dim),
            ],
            name = 'encoder'
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=2**2*64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(-1, 1, 2**2*64)),
                tf.keras.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    activation='relu'),
                #BatchNormalization(),

                tf.keras.layers.Conv2DTranspose(
                    filters=32,
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

        plt.figure(figsize=(n*2+1, n))
        plt.imshow(canvas_orig[:,:,0], origin="upper", cmap='gray')
        plt.draw()
        plt.show()
        
        # canvas_orig = np.empty((imd*n , imd * n, 1))
        # for i in range(n):
        #     g = preds[i*n:i*n+n]

        #     for j in range(n):
        #         canvas_orig[i * imd:(i + 1) * imd, j * imd:(j + 1) * imd] = \
        #             g[j].reshape([imd, imd, 1])
        # # print(min(canvas_orig))
        # # print(max(canvas_orig))
        # plt.figure(figsize=(n, n))
        # plt.imshow(canvas_orig[:,:,0], origin="upper", cmap='gray') #, vmin=min(canvas_orig), vmax=max(canvas_orig))
        # plt.draw()
        # plt.show()
        



#@tf.function
def compute_loss(vae_generator, discriminator, x, beta, gamma, gen_dec_weight):
    mean, logvar = vae_generator.encode(x)
    z = vae_generator.reparameterize(mean, logvar)
    y = vae_generator.decoder(z)
    z2 = tf.random.normal(shape=z.shape)
    y2 = vae_generator.decoder(z2)

    #print(y.shape)
    #print(x.shape)

    

    real_output = discriminator(x, training=True)
    fake_output = discriminator(tf.sigmoid(y), training=True)
    fake_output2 = discriminator(tf.sigmoid(y2), training=True)

    gen_loss = generator_loss(fake_output, fake_output2)
    disc_loss = discriminator_loss(real_output, fake_output, fake_output2)



    rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x), axis=[1,2,3])
    # print(tf.math.reduce_max(rec_loss))
    rec_loss = tf.reduce_mean(rec_loss)
    # y = tf.sigmoid(y)
    # #y2 = tf.sigmoid(y2)
    # rec_loss = tf.reduce_sum(tf.math.square(x - y), axis=[1,2,3])
    # rec_loss = tf.reduce_mean(rec_loss)




    kl_loss = -0.5*tf.math.reduce_sum((1 + logvar - tf.square(mean) - tf.exp(logvar)), axis=1)
    #kl_loss *= 0
    # https://openreview.net/forum?id=Sy2fzU9gl
  
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= beta

    # print('kl', kl_loss)
    # print('recl', rec_loss)
    # print('gen_l', gen_loss)
    # print('discl', disc_loss)

    #disc_loss *= 0.5


    en_loss = rec_loss+kl_loss
    de_loss = gamma*rec_loss+gen_loss*gen_dec_weight

    return en_loss, de_loss, disc_loss, {'kl': kl_loss.numpy(), 'recl': rec_loss.numpy(), 
                                         'gen_l': gen_loss.numpy(), 'discl': disc_loss.numpy()}


def discriminator_loss(real_output, fake_output, fake_output2):

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    fake_loss2 = cross_entropy(tf.zeros_like(fake_output2), fake_output2)

    total_disc_loss = real_loss + (fake_loss + fake_loss2)*0.5
    return total_disc_loss



def generator_loss(fake_output, fake_output2):
    loss1 = cross_entropy(tf.ones_like(fake_output), fake_output)
    loss2 = cross_entropy(tf.ones_like(fake_output2), fake_output2)
    return loss1 + loss2


if __name__=='__main__':

    input_shape = (28, 28, 1)
    disc_layers = [
        InputLayer(input_shape=input_shape),
        Conv2D(16, 4, strides=2, padding='same'),
        Dropout(0.5),
        LeakyReLU(),
        Conv2D(32, 4, strides=2, padding='same'),
        Dropout(0.5),
        LeakyReLU(),
        # Conv2D(32, 4, strides=2, padding='same'),
        # Dropout(0.5),
        # LeakyReLU(),
        #Conv2D(64, 4, strides=2, padding='same'),
        Flatten(),
        Dense(1)
    ]

    discriminator = tf.keras.Sequential(disc_layers, name="discriminator")
    discriminator.summary()

    generator = VAE(input_shape=input_shape, latent_dim=64)
    generator.summary()

    encoder_optimizer = tf.keras.optimizers.Adam(1e-4)
    decoder_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    # for test_images, labels in test_ds:
    #     #generator.test(test_images)
    #     break


    beta = tf.Variable(1.)
    gamma = tf.Variable(1.)
    gen_dec = tf.Variable(0.)
    #gamma = tf.Variable(0.1)
    #print(dir(train_ds))
    for e in range(100):
        print(e)
        st = time()


        for i, (images, labels) in enumerate(tqdm(train_ds)):
            #print(i)
            # if not (i+1)%234:
            #     #print('hello')
            #     for test_images, labels in test_ds:
            #         generator.test(test_images)
            #         break
            images = tf.dtypes.cast(images, tf.float32)

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3: 
                en_loss, de_loss, disc_loss, info = compute_loss(generator, discriminator, images, beta, gamma, gen_dec)

            if not (i+1)%90:
                print(i, info)

            gradients_of_encoder = tape1.gradient(en_loss, generator.encoder.trainable_variables)
            gradients_of_decoder = tape2.gradient(de_loss, generator.decoder.trainable_variables)


            encoder_optimizer.apply_gradients(zip(gradients_of_encoder, generator.encoder.trainable_variables))
            decoder_optimizer.apply_gradients(zip(gradients_of_decoder, generator.decoder.trainable_variables))
            
            

            #if e > 2:
            #    gen_dec = tf.Variable(5.)
            gradients_of_discriminator = tape3.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        print(time()-st)

        for test_images, labels in test_ds:
            generator.test(test_images)
            break
