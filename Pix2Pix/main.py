"""
Implementation of:
https://arxiv.org/pdf/1611.07004.pdf
"""

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Concatenate, Input, ZeroPadding2D

import os
from time import time
import matplotlib.pyplot as plt

from tqdm import tqdm

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

OUTPUT_CHANNELS = 3

LAMBDA = 100




def load(image_file):
	image = tf.io.read_file(image_file)
	image = tf.image.decode_jpeg(image)

	w = tf.shape(image)[1]

	w = w // 2
	real_image = image[:, :w, :]
	input_image = image[:, w:, :]

	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)

	return input_image, real_image



def resize(input_image, real_image, height, width):
	input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return input_image, real_image

def random_crop(input_image, real_image):
	stacked_image = tf.stack([input_image, real_image], axis=0)
	cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

	return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
	input_image = (input_image / 127.5) - 1
	real_image = (real_image / 127.5) - 1

	return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
	# resizing to 286 x 286 x 3
	input_image, real_image = resize(input_image, real_image, 286, 286)

	# randomly cropping to 256 x 256 x 3
	input_image, real_image = random_crop(input_image, real_image)

	if tf.random.uniform(()) > 0.5:
		# random mirroring
		input_image = tf.image.flip_left_right(input_image)
		real_image = tf.image.flip_left_right(real_image)

	return input_image, real_image

# plt.figure(figsize=(6, 6))
# for i in range(4):
# 	rj_inp, rj_re = random_jitter(inp, re)
# 	plt.subplot(2, 2, i+1)
# 	plt.imshow(rj_inp/255.0)
# 	plt.axis('off')
# plt.show()
# raise

def load_image_train(image_file):
	input_image, real_image = load(image_file)
	input_image, real_image = random_jitter(input_image, real_image)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image

def load_image_test(image_file):
	input_image, real_image = load(image_file)
	input_image, real_image = resize(input_image, real_image,
	                               IMG_HEIGHT, IMG_WIDTH)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image




class DownSample(tf.keras.Model):

	def __init__(self, filters, size, apply_batchnorm=True):
		super().__init__()


		initializer = tf.random_normal_initializer(0., 0.02)
		self.apply_batchnorm = apply_batchnorm

		self.conv_layer = Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)

		if apply_batchnorm:
			self.bn_layer = BatchNormalization()

		self.act = LeakyReLU()
		


	def call(self, x):
		x = self.conv_layer(x)
		if self.apply_batchnorm:
			x = self.bn_layer(x)
		x = self.act(x)
		return x


# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print (down_result.shape)
class UpSample(tf.keras.Model):

	def __init__(self, filters, size, apply_dropout=False, **kwargs):
		super().__init__(**kwargs)


		initializer = tf.random_normal_initializer(0., 0.02)
		self.apply_dropout = apply_dropout

		self.conv_layer = Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)

		self.bn_layer = BatchNormalization()

		if apply_dropout:
			self.do_layer = Dropout(0.5)

		self.act = LeakyReLU()


	def call(self, x, **kwargs):
		x = self.conv_layer(x, **kwargs)
		x = self.bn_layer(x, **kwargs)
		if self.apply_dropout:
			x = self.do_layer(x, **kwargs)

		x = self.act(x, **kwargs)
		return x


class Generator(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.down_stack = [
			DownSample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
			DownSample(128, 4), # (bs, 64, 64, 128)
			DownSample(256, 4), # (bs, 32, 32, 256)
			DownSample(512, 4), # (bs, 16, 16, 512)
			DownSample(512, 4), # (bs, 8, 8, 512)
			DownSample(512, 4), # (bs, 4, 4, 512)
			DownSample(512, 4), # (bs, 2, 2, 512)
			DownSample(512, 4), # (bs, 1, 1, 512)
		]

		self.up_stack = [
			UpSample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
			UpSample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
			UpSample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
			UpSample(512, 4), # (bs, 16, 16, 1024)
			UpSample(256, 4), # (bs, 32, 32, 512)
			UpSample(128, 4), # (bs, 64, 64, 256)
			UpSample(64, 4), # (bs, 128, 128, 128)
		]


		initializer = tf.random_normal_initializer(0., 0.02)
		self.last = Conv2DTranspose(OUTPUT_CHANNELS, 4,
		                                     strides=2,
		                                     padding='same',
		                                     kernel_initializer=initializer,
		                                     activation='tanh') # (bs, 256, 256, 3)

		self.concat = Concatenate()

		#self.inputs = Input(shape=[None,None,3])

	def call(self, x, **kwargs):

		# Downsampling through the model
		skips = []
		for down in self.down_stack:
			x = down(x, **kwargs)
			skips.append(x)

		skips = reversed(skips[:-1])

		# Upsampling and establishing the skip connections
		for up, skip in zip(self.up_stack, skips):
			x = up(x, **kwargs)
			x = self.concat([x, skip])

		x = self.last(x, **kwargs)

		return x

class Discriminator(tf.keras.Model):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		initializer = tf.random_normal_initializer(0., 0.02)

		#inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
		#tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

		self.concat = Concatenate()


		# x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
		seq_layers = [

			DownSample(64, 4, False), # (bs, 128, 128, 64)
			DownSample(128, 4), # (bs, 64, 64, 128)
			DownSample(256, 4), # (bs, 32, 32, 256)

			ZeroPadding2D(),  # (bs, 34, 34, 256)
			Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False), # (bs, 31, 31, 512)

			BatchNormalization(),

			LeakyReLU(),

			ZeroPadding2D(), # (bs, 33, 33, 512)

			Conv2D(1, 4, strides=1, kernel_initializer=initializer) # (bs, 30, 30, 1)
		]
		self.seq = tf.keras.Sequential(seq_layers)

	def call(self, x, **kwargs):

		x = self.concat(x)
		x = self.seq(x, **kwargs)
		return x



def discriminator_loss(disc_real_output, disc_generated_output, loss_object):
	real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

	generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

	total_disc_loss = real_loss + generated_loss

	return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, loss_object):
	gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

	# mean absolute error
	l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

	total_gen_loss = gan_loss + (LAMBDA * l1_loss)

	return total_gen_loss



def generate_images(model, test_input, tar):
	# the training=True is intentional here since
	# we want the batch statistics while running the model
	# on the test dataset. If we use training=False, we will get
	# the accumulated statistics learned from the training dataset
	# (which we don't want)
	prediction = model(test_input, training=True)
	plt.figure(figsize=(15,15))

	display_list = [test_input[0], tar[0], prediction[0]]
	title = ['Input Image', 'Ground Truth', 'Predicted Image']

	for i in range(3):
		plt.subplot(1, 3, i+1)
		plt.title(title[i])
		# getting the pixel values between [0, 1] to plot it.
		plt.imshow(display_list[i] * 0.5 + 0.5)
		plt.axis('off')
	plt.show()


@tf.function
def train_step(input_image, target, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer):
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		gen_output = generator(input_image, training=True)

		disc_real_output = discriminator([input_image, target], training=True)
		disc_generated_output = discriminator([input_image, gen_output], training=True)

		gen_loss = generator_loss(disc_generated_output, gen_output, target, loss_object)
		disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

	generator_gradients = gen_tape.gradient(gen_loss,
	                                  generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss,
	                                       discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_gradients,
	                                  generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
	                                      discriminator.trainable_variables))


def fit(train_ds, epochs, test_ds, generator, discriminator):

	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	                                 discriminator_optimizer=discriminator_optimizer,
	                                 generator=generator,
	                                 discriminator=discriminator)

	#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


	loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	for epoch in range(epochs):
		print('E', epoch)
		start = time()

		# Train
		for i, (input_image, target) in enumerate(tqdm(train_ds)):
			train_step(input_image, target, generator, discriminator, loss_object, generator_optimizer, discriminator_optimizer)

		# Test on the same image so that the progress of the model can be 
		# easily seen.
		for example_input, example_target in test_ds.take(1):
			generate_images(generator, example_input, example_target)

		# saving (checkpoint) the model every 20 epochs
		if (epoch + 1) % 20 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time()-start))




if __name__=='__main__':
	

	inp, re = load(PATH+'train/100.jpg')
	# # casting to int for matplotlib to show the image
	# plt.figure()
	# plt.imshow(inp/255.0)
	# plt.figure()
	# plt.imshow(re/255.0)
	# plt.show()

	# raise

	generator = Generator()

	gen_output = generator(inp[tf.newaxis,...], training=False)
	plt.imshow(gen_output[0,...])
	plt.show()


	discriminator = Discriminator()
	disc_out = discriminator([inp[tf.newaxis,...], gen_output], training=False)
	plt.imshow(disc_out[0,...,-1], vmin=-20, vmax=20, cmap='RdBu_r')
	plt.colorbar()
	plt.show()


	train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
	train_dataset = train_dataset.map(load_image_train,
	                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
	train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
	train_dataset = train_dataset.batch(1)

	test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
	test_dataset = test_dataset.map(load_image_test)
	test_dataset = test_dataset.batch(1)



	# EPOCHS = 150

	# fit(train_dataset, EPOCHS, test_dataset, generator, discriminator)

	# Run the trained model on the entire test dataset

	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	                                 discriminator_optimizer=discriminator_optimizer,
	                                 generator=generator,
	                                 discriminator=discriminator)
	checkpoint.restore('/home/sam/DL_big_files/Pix2Pix/ckpt-7')

	for inp, tar in test_dataset.take(5):
		generate_images(generator, inp, tar)