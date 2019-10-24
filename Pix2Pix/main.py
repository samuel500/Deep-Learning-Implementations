"""
Implementation of:
https://arxiv.org/pdf/1611.07004.pdf
"""

import tensorflow as tf

import os
from time import time
import matplotlib.pyplot as plt

from tqdm import tqdm

from pix2pix_models import *
from pix2pix_image_utils import *

_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')


BUFFER_SIZE = 400
BATCH_SIZE = 1


LAMBDA = 100



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

	generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
	discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


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
	train_dataset = train_dataset.batch(BATCH_SIZE)

	test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
	test_dataset = test_dataset.map(load_image_test)
	test_dataset = test_dataset.batch(BATCH_SIZE)



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