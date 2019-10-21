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


import imageio


import scipy.ndimage as nd



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



def calc_loss(img, model, target=None, channels=None):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    #print('ims', img.shape)

    img_batch = tf.expand_dims(img, axis=0)
    #img_batch = img
    layer_activations = model(img_batch) #.numpy()

    #print(layer_activations[0].shape)
    
    #print(layer_activations)
    #print('Predicted:', iv3.decode_predictions(layer_activations, top=5)[0])

    if target is not None:
        layer_activations = layer_activations[0]
        print('las', layer_activations.shape)
        print('ts', target.shape)
        ch = target.shape[-1]
        x = tf.reshape(layer_activations, (ch,-1))
        y = tf.reshape(target, (ch,-1))
        print('xs', x.shape)
        print('ys', y.shape)
        A = tf.matmul(x, y, transpose_a=True)
        print('As', A.shape)
        #raise
        idx1 = np.array(range(x.shape[0]))
        idx2 = tf.argmax(A, axis=0).numpy() #[:x.shape[1]]
        Aa = np.array(A)

        print('amax Aa', np.amax(Aa))
        print('mean Aa', np.mean(Aa))

        print(idx1.shape)
        print(idx2.shape)

        s = list(zip(list(idx1), list(idx2)))
        result = tf.gather_nd(x, s)
        print(result.shape)

        #diff = y[:,list(tf.argmax(A, axis=1).numpy())]
        #return tf.math.reduce_mean(A)
        return tf.math.reduce_mean(result)


    elif channels is not None:
   
        if not hasattr(channels, '__iter__'):
            channels = [channels]

        C = layer_activations[0].shape[-1]

        if max(channels) >= C:
            raise IndexError("Invalid channel index:{} (max:{})".format(max(channels),C-1))

        t = tf.gather(layer_activations[0], axis=3, batch_dims=0, indices=list(channels))

        return tf.math.reduce_mean(t)

    else:
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


def zoom_in():
    pass

#@tf.function
def get_tiled_gradients(model, img, tile_size=512, target=None, channels=None):
    shift_down, shift_right, img_rolled = random_roll(img, tile_size)

    # Initialize the image gradients to zero.
    gradients = tf.zeros_like(img_rolled)

    tot_loss = 0.

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
                if min(img_tile.shape[:2]) < 15: # keras model does not accept inputs that are too small
                    continue    
                else:
                    loss = calc_loss(img_tile, model, target, channels)
                    if tf.math.is_nan(loss):
                        continue
                    tot_loss += float(loss) * np.prod(img_tile.shape[:2])/np.prod(img_rolled.shape[:2]) # add loss to total, weighted with size of tile

            # Update the image gradients for this tile.
            gradients += tape.gradient(loss, img_rolled)

    print('tot_loss:', tot_loss)
    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

    # Normalize the gradients.
    #gradients /= tf.math.reduce_std(gradients) + 1e-8
    #print('std', tf.math.reduce_std(gradients))
    #print('mean', np.abs(gradients).mean())
    gradients /= np.abs(gradients).mean() + 1e-8
    return gradients


def deep_dream(model, img, steps_per_octave=100, step_size=0.01,
                                num_octaves=4, octave_scale=1.4, target=None, channels=None, zoom=1, create_gif=False):
    #img = tf.keras.preprocessing.image.img_to_array(img)
    #img = iv3.preprocess_input(img)
    img = model.preprocess_image(img)


    if create_gif:
        gif_file =  'test' + str(np.random.randint(10000))  + '.gif'
        writer = imageio.get_writer(gif_file, mode='I', duration=0.1)

    for octave in range(num_octaves):
        # Scale the image based on the octave
        if octave>0:
            new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32)*octave_scale
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in range(steps_per_octave):


            gradients = get_tiled_gradients(model, img, target=target, channels=channels)
            img = img + gradients*step_size
            img = model.clip_image(img)

            if zoom is not 1:
                img_y, img_x = img.shape[:2]

                new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32)*zoom
                img = tf.image.resize(img, tf.cast(new_size, tf.int32))

                img = tf.image.resize_with_crop_or_pad(img, img_y, img_x)
                #img = img[(new_size[0]-img_y)//2:(new_size[0]-img_y)//2+img_y,
                #    (new_size[1]-img_x)//2:(new_size[1]-img_x)//2+img_x,:]

            if create_gif:
                img_gif = model.deprocess_image(img, rescale=True)
                writer.append_data(img_gif)
   

            if step % 1000 == 0:
                show(model.deprocess_image(img))
                print("Octave {}, Step {}".format(octave, step))
        show(model.deprocess_image(img))

        print("Octave {}, Step {}".format(octave, step))

    if create_gif:
        writer.close()

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

    if len(layers) == 1:
        layers = layers[0]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)


    dream_model.preprocess_image = model_class.preprocess_input
    dream_model.deprocess_image = deprocess
    dream_model.clip_image = lambda img: tf.clip_by_value(img, -1., 1.)

    return dream_model


if __name__=='__main__':

    # TODO:
    # Target activations
    # Neuron activations
    # Zoom in


    original_img = load_image('private/greg.jpg', size=448)
    #original_img = load_image('sky.jpg', size=384)
    #original_img = load_image('blackpool.jpg', size=512)

    #original_img = np.random.uniform(low=0., high=1., size=original_img.shape)

    print(original_img.shape)


    #show(original_img)

    #shift_down, shift_right, img_rolled = random_roll(np.array(original_img), 512)
    #print(shift_down, shift_right)
    #show(img_rolled)


    content_layers = [12]
    dream_model = get_squeezenet_model(content_layers)



    target_img = load_image('flowers.jpg', size=240)
    target_img = dream_model.preprocess_image(target_img)
    target = dream_model(target_img[None])[0]
    #print(target.shape)
    target = None

    #names = ['mixed2'] #['predictions'] # #, 'mixed9'] #['mixed3', 'mixed5'] #['mixed2']
    #names = ['add_11']
    #names = ['add_2']
    #from tensorflow.keras.applications import inception_v3 as iv3
    #from tensorflow.keras.applications import xception as xce
    #from tensorflow.keras.applications import nasnet
    #dream_model = get_keras_model(names, model_class=nasnet, show_summary=True)

    #channels = range(295, 300) # [28] #range(300, 305)
    channels = None

    dream_img = deep_dream(model=dream_model, img=original_img, step_size=0.15, 
            steps_per_octave=200, target=target, channels=channels, zoom=1.06,
            num_octaves=1, octave_scale=1.3, create_gif=True)



    _, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Original Img.')
    axarr[1].set_title('Dream Img.')
    axarr[0].imshow(original_img)
    axarr[1].imshow(dream_img)
    plt.show()
    plt.figure()
