import sys
sys.path.append("..") # to import things one level up

import tensorflow as tf

import numpy as np
from scipy.misc import imread, imresize

import imageio

from matplotlib import pyplot as plt

import imageio

from tqdm import tqdm
import cv2

import scipy.ndimage as nd

from image_utils import load_image, show

from models_utils import *



def calc_loss(img, model, target=None, channels=None):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    #print('ims', img.shape)

    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch) #.numpy()

    #print(layer_activations[0].shape)
    
    #print('Predicted:', iv3.decode_predictions(layer_activations, top=5)[0])

    if target is not None:
        
        raise NotImplementedError

        '''
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
        idx1 = np.array(range(A.shape[0]))
        idx2 = tf.argmax(A, axis=1).numpy() #[:x.shape[1]]
        #Aa = np.array(A)

        #print('amax Aa', np.amax(Aa))
        #print('mean Aa', np.mean(Aa))

        print(idx1.shape)
        print(idx2.shape)

        s = list(zip(list(idx1), list(idx2)))
        result = tf.gather_nd(A, s)
        print(result.shape)

        #diff = y[:,list(tf.argmax(A, axis=1).numpy())]
        #return tf.math.reduce_mean(A)
        return tf.math.reduce_mean(result)
        '''

    elif channels is not None:

        if not hasattr(channels, '__iter__'):
            channels = [channels]

        C = layer_activations[0].shape[-1]

        if max(channels) >= C:
            raise IndexError("Invalid channel index:{} (max:{})".format(max(channels),C-1))

        if len(layer_activations) == 5:
            layer_activations = layer_activations[0]
        t = tf.gather(layer_activations, axis=3, batch_dims=0, indices=list(channels))

        return tf.math.reduce_mean(t)

    else:
        losses = []
        if type(layer_activations) is list:
            for act in layer_activations:
                loss = tf.math.reduce_mean(act)
                losses.append(loss)
        else:
            loss = tf.math.reduce_mean(layer_activations)
            losses.append(loss)

        return tf.reduce_sum(losses)


#@tf.function
def get_tiled_gradients(model, img, tile_size=1024, target=None, channels=None):
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

    #print('tot_loss:', tot_loss)
    # Undo the random shift applied to the image and its gradients.
    gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

    # Normalize the gradients.
    #gradients /= tf.math.reduce_std(gradients) + 1e-8
    #print('std', tf.math.reduce_std(gradients))
    #print('mean', np.abs(gradients).mean())
    gradients /= tf.math.reduce_mean(tf.math.abs(gradients)) + 1e-8
    return gradients


def deep_dream(model, img, steps_per_octave=10, step_size=0.01,
                                num_octaves=4, octave_scale=1.3, target=None, channels=None, zoom=1, 
                                create_gif=False, create_video=False, rand_channel=None,
                                loop_channel=None):
    """
    Inputs:
    :model: model to extract layer activation
    :img: starting image
    :steps_per_octave: number of alteration steps for each octave
    :step_size: size of each gradient alteration
    :num_octaves: number of scale changes
    :octave_scale: image size change per octave
    :channels: channels to maximize
    :zoom: size of zoom to apply per training step (1 = no zoom)
    :create_gif: create gif of training
    :create_video: create video of training
    :rand_channel: instruction dict to randomly change the channels to maximize at regular interval (for variation)
    """

    img = model.preprocess_image(img)


    if create_gif:
        gif_file =  'deep_dream' + str(np.random.randint(10000))  + '.gif'
        gif_writer = imageio.get_writer(gif_file, mode='I', duration=0.2)
    if create_video:
        height, width = img.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"X264") 

        video_writer = cv2.VideoWriter('video'+str(np.random.randint(10000))+'.avi', fourcc, 18.,(width,height))


    for octave in range(num_octaves):
        # Scale the image based on the octave
        if octave>0:
            new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32)*octave_scale
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

        for step in tqdm(range(steps_per_octave)):

            if rand_channel is not None:
                if not step%rand_channel['every']:
                    channels = np.random.choice(rand_channel['choices'], size=rand_channel['size'], replace=False)
                    if 'verbose' in rand_channel:
                        if rand_channel['verbose']:
                            print('channels @ step', step, ':', sorted(channels))

            if loop_channel is not None:
                if channels is None:
                    channels = [0]
                if not step%loop_channel['every']:
                    if step:
                        channels[0] += 1
                    if 'verbose' in loop_channel:
                        if loop_channel['verbose']:
                            print('channels @ step', step, ':', sorted(channels))

            gradients = get_tiled_gradients(model, img, target=target, channels=channels)
            img = img + gradients*step_size
            img = model.clip_image(img)

            if zoom is not 1:
                img_y, img_x = img.shape[:2]

                new_size = tf.cast(tf.convert_to_tensor(img.shape[:2]), tf.float32)*zoom
                img = tf.image.resize(img, tf.cast(new_size, tf.int32))

                img = tf.image.resize_with_crop_or_pad(img, img_y, img_x)


            if create_gif:
                if not step%6:
                    img_gif = model.deprocess_image(img)
                    gif_writer.append_data(img_gif)

            if create_video:

                img_v = model.deprocess_image(img)

                if loop_channel is not None:
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, img_v.shape[0]-20)
                    fontScale              = 1
                    fontColor              = (0,0,0)
                    lineType               = 2

                    img_v = cv2.putText(img_v, str(channels[0]), 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)


                img_video = cv2.cvtColor(img_v, cv2.COLOR_RGB2BGR)
                video_writer.write(img_video)

            if not (step+1)%20000:
                show(model.deprocess_image(img))
                print("Octave {}, Step {}".format(octave, step))

        #show(model.deprocess_image(img))

        print("Octave {}, Step {}".format(octave, step))

    if create_gif:
        gif_writer.close()

    if create_video:
        cv2.destroyAllWindows()
        video_writer.release()


    result = model.deprocess_image(img)

    return result


def random_roll(img, maxroll):
    # Randomly shift the image to avoid tiled boundaries, and jitter regularization
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    shift_down, shift_right = shift[0],shift[1]
    img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
    return shift_down, shift_right, img_rolled



if __name__=='__main__':


    #original_img = load_image('private/greg.jpg', size=512)
    #original_img = load_image('sky.jpg', size=384)
    original_img = load_image('blackpool.jpg', size=512)
    #print(original_img.shape)


    #original_img = np.random.uniform(low=0., high=255., size=(447, 712, 3))

    print(original_img.shape)

    #show(original_img)


    #content_layers = [10]
    #dream_model = get_squeezenet_model(content_layers)


    #target_img = load_image('flowers.jpg', size=240)
    #target_img = dream_model.preprocess_image(target_img)
    #target = dream_model(target_img[None])[0]
    #print(target.shape)


    #names = ['mixed8'] #['predictions'] # #, 'mixed9'] #['mixed3', 'mixed5'] #['mixed2']
    #names = ['add_11']
    #names = ['add_2']
    #names = ['normal_concat_7']

    names = ['block_15_add'] #['block_9_add'] #['Conv_1_bn'] #['block_13_project_BN'] #['block_12_add'] #['block_16_project'] #['block_15_add']
    #from tensorflow.keras.applications import inception_v3 as iv3
    from tensorflow.keras.applications import mobilenet_v2 as mb2
    #from tensorflow.keras.applications import xception as xce
    #from tensorflow.keras.applications import nasnet

    dream_model = get_keras_model(names, model_class=mb2, model_i=0, show_summary=False)
    #channels = range(50, 60) # [28] #range(300, 305)
    channels = None
    #raise
    rand_channel = {
        'every': 200,
        'size': 1,
        'choices': range(96),
        'verbose': True
    }
    rand_channel = None

    loop_channel = {
        'every': 180,
        'choices':range(160),
        'verbose': True
    }
    #loop_channel = None

    dream_img = deep_dream(model=dream_model, img=original_img, step_size=0.12, 
            steps_per_octave=28800, channels=channels, zoom=1.012,
            num_octaves=1, octave_scale=1.3, create_gif=False, create_video=True,
            rand_channel=rand_channel, loop_channel=loop_channel)



    _, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Original Img.')
    axarr[1].set_title('Dream Img.')
    axarr[0].imshow(original_img)
    axarr[1].imshow(dream_img)
    plt.show()
    plt.figure()
