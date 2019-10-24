import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout, Concatenate, Input, ZeroPadding2D



OUTPUT_CHANNELS = 3


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

