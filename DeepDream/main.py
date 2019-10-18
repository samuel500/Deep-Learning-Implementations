import sys
sys.path.append("..") # to import things one level up

import tensorflow as tf

import numpy as np
from scipy.misc import imread, imresize

import imageio

from matplotlib import pyplot as plt

from tensorflow.keras.preprocessing import image


from image_utils import load_image
from squeezenet import SqueezeNet
from utils import weight_to_weight, get_ckpt_weights



# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)


# Display an image
def show(img):
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.axis('off')
    plt.imshow(img)
    plt.show()



def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    #img_batch = img
    layer_activations = model(img_batch) #.numpy()
    #print(layer_activations)
    #print('Predicted:', iv3.decode_predictions(layer_activations, top=5)[0])


    losses = []
    if type(layer_activations) is list:
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)
    else:
        #t = np.zeros(1000)
        #t = layer_activations
        #t[0][594] = 1

        loss = tf.math.reduce_mean(layer_activations)
        #loss = tf.math.reduce_mean(layer_activations)
        losses.append(loss)
    return tf.reduce_sum(losses)


#@tf.function
def get_tiled_gradients(model, img, tile_size=512):
    shift_down, shift_right, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    for x in tf.range(0, img_rolled.shape[0], tile_size):
        for y in tf.range(0, img_rolled.shape[1], tile_size):
            # Calculate the gradients for this tile.
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img_rolled`.
                # `GradientTape` only watches `tf.Variable`s by default.
                tape.watch(img_rolled)

                # Extract a tile out of the image.
                #print(img_rolled.shape)
                #print(x+tile_size)
                img_tile = img_rolled[x:x+tile_size, y:y+tile_size]
                print(img_tile.shape)
                if min(img_tile.shape[:2]) < 15: # model does not accept inputs that are too small
                    print('wowowow')
                    pass    
                else:
                    loss = calc_loss(img_tile, model)
                    print(loss)

            if min(img_tile.shape[:2]) < 15:
                print('wowowo')
                pass    
            else:
                # Update the image gradients for this tile.
                gradients += tape.gradient(loss, img_rolled)

    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

    # Normalize the gradients.
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    #print('std', tf.math.reduce_std(gradients))
    #print('mean', np.abs(gradients).mean())
    #gradients /= np.abs(gradients).mean() + 1e-8
    return gradients


def deep_dream(model, img, steps_per_octave=100, step_size=0.01,
                                num_octaves=4, octave_scale=1.3):
    #img = tf.keras.preprocessing.image.img_to_array(img)
    #img = iv3.preprocess_input(img)
    img = model.preprocess_image(img)

    for octave in range(num_octaves):
        # Scale the image based on the octave
        if octave>0:
            new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32)*octave_scale
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(model, img)
            img = img + gradients*step_size
            img = model.clip_image(img)

            if step % 50 == 0:
                show(model.deprocess_image(img))
                print("Octave {}, Step {}".format(octave, step))

    result = model.deprocess_image(img)
    show(result)

    return result


def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries, and jitter regularization
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0],shift[1]
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled



def get_squeezenet_model(layers):

    base_model = SqueezeNet(content_layers)

    all_n = []
    
    all_n = get_ckpt_weights('../squeezenet_weights/squeezenet.ckpt')
    all_weights = []
    for w in base_model.model.weights:
        all_weights.append(all_n[weight_to_weight[w.name]])
    base_model.set_weights(all_weights)
    base_model.trainable = False

    dream_model = base_model 

    return dream_model


def get_keras_model(names, model_class, show_summary=False):
    #base_model = iv3.InceptionV3(include_top=True, weights='imagenet')

    print(dir(model_class))
    base_model = getattr(model_class, dir(model_class)[0])(include_top=True, weights='imagenet')
    if show_summary:
        base_model.summary()
    # Maximize the activations of these layers


    layers = [base_model.get_layer(name).output for name in names]
    #print(layers)
    if len(layers) == 1:
        layers = layers[0]
    # Create the feature extraction model
    #print(base_model.model.input)
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    dream_model.preprocess_image = iv3.preprocess_input
    dream_model.deprocess_image = deprocess
    dream_model.clip_image = lambda img: tf.clip_by_value(img, -1., 1.)

    return dream_model


if __name__=='__main__':


    #original_img = load_image('private/greg.jpg', size=448)
    #original_img = load_image('sky.jpg', size=512)
    original_img = load_image('blackpool.jpg', size=512)

    original_img = np.random.uniform(low=0., high=1., size=original_img.shape)

    print(original_img.shape)

    #show(original_img)

    #shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
    #print(shift_down, shift_right)
    #show(img_rolled)


    content_layers = [6]
    dream_model = get_squeezenet_model(content_layers)

    #names = ['mixed2'] #['predictions'] # #, 'mixed9'] #['mixed3', 'mixed5'] #['mixed2']
    #names = ['add_11']
    #names = ['add_2']
    #from tensorflow.keras.applications import inception_v3 as iv3
    #from tensorflow.keras.applications import xception as xce
    #from tensorflow.keras.applications import nasnet
    #dream_model = get_keras_model(names, model_class=nasnet, show_summary=True)


    dream_img = deep_dream(model=dream_model, img=original_img, step_size=0.025, steps_per_octave=20)


    _, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Original Img.')
    axarr[1].set_title('Dream Img.')
    axarr[0].imshow(original_img)
    axarr[1].imshow(dream_img)
    plt.show()
    plt.figure()
