"""
Implementation of:
https://arxiv.org/pdf/1508.06576.pdf
"""

import sys
sys.path.append("..") # to import things one level up

import tensorflow as tf 


import numpy as np 

from pprint import pprint
from copy import deepcopy
import matplotlib.pyplot as plt

from PIL import Image
import imageio

from image_utils import load_image
from squeezenet import SqueezeNet
from utils import weight_to_weight, get_ckpt_weights




def style_transfer(content_image, style_image, content_layers, content_weight,
                   style_layers, style_weights, total_variation_weight, model, 
                   init_random=False, create_gif=False, save_images=True, show_interval=50,
                   max_iter=200, image_size=512, lr=0.1):
    """
    Inputs:
    :content_image: filename of content image
    :style_image: filename of style image
    :content_layer: layer to use for content loss
    :content_weight: weighting on content loss
    :style_layers: list of layers to use for style loss
    :style_weights: list of weights to use for each layer in style_layers
    :total_variation_weight: weight of total variation regularization loss term
    :init_random: initialize the starting image to uniform random noise
    :create_gif: create a gif of the training progress
    :save_images: save styled image at interval
    :show_interval: display styled image at iteration interval
    :max_iter: stop training after max_iter training iterations
    :image_size: size to rescale smallest image dimension to
    """

    style_t_name = content_image.split('/')[-1].split('.')[0] + '_' + style_image.split('/')[-1].split('.')[0]  

    content_img = model.preprocess_image(load_image(content_image, image_size))
    feats = model.get_layers(content_img[None])
    content_targets = [feats[i] for i in content_layers]

    # Extract features from the style image
    style_img = model.preprocess_image(load_image(style_image, image_size))
    feats = model.get_layers(style_img[None])
    style_feat_vars = [feats[i] for i in style_layers]

    style_target_vars = []
    # Compute Gram matrix
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    style_targets = style_target_vars


    # Initialize generated image to content image
    
    if init_random:
        img_var = tf.Variable(tf.random_uniform_initializer(minval=0, maxval=1)(content_img[None].shape))
    else:
        img_var = tf.Variable(content_img[None])

    

    lr_var = tf.Variable(lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_var)

    img_var.assign(model.clip_image(img_var))
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(model.deprocess_image(content_img))
    axarr[1].imshow(model.deprocess_image(style_img))
    plt.show()
    plt.figure()

    if create_gif:
        gif_file = style_t_name + str(np.random.randint(10000))  + '.gif'
        writer = imageio.get_writer(gif_file, mode='I', duration=0.18)

    for t in range(max_iter):

        with tf.GradientTape() as tape:

            tape.watch(img_var)
            feats = model.get_layers(img_var)
            content_feats = [feats[i] for i in content_layers]

            # Compute loss
            c_loss = content_loss(content_weight, content_feats, content_targets)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = total_variation_loss(img_var, total_variation_weight)
            print('content_loss', c_loss)
            print('style_loss', s_loss)
            print('total_variation_loss', t_loss)

            loss =  s_loss + t_loss + c_loss
            print(t, ' Loss:', loss)

        gradients = tape.gradient(loss, [img_var])

        optimizer.apply_gradients(zip(gradients, [img_var]))


        img_var.assign(model.clip_image(img_var))

        if not t % show_interval:
            print('Iteration {}'.format(t))
            image = model.deprocess_image(img_var[0], rescale=True)
            plt.imshow(image)
            plt.axis('off')
            plt.show()

            if save_images:
                im = Image.fromarray(image)
                im.save(style_t_name + "_styled{}_t:{}.jpg".format(np.random.randint(10000), t))

        if create_gif:
            if not t % 16: 
                image = model.deprocess_image(img_var[0], rescale=True)
                #image = Image.fromarray(image)
                writer.append_data(image)
    if create_gif:
        writer.close()

    image = model.deprocess_image(img_var[0], rescale=True)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
 
    im = Image.fromarray(image)
    im.save(style_t_name + "_styled{}_t:{}.jpg".format(np.random.randint(10000), t))


@tf.function
def content_loss(content_weight, content_outputs, content_targets):
    c_loss = tf.add_n([tf.reduce_mean(tf.square(content_outputs[i]-content_targets[i])) 
                             for i in range(len(content_outputs))])
    c_loss *= content_weight/len(content_outputs)
    return c_loss


def gram_matrix(features, normalize=True):
    shape = tf.shape(features)
    H, W, C = shape[1], shape[2], shape[3]
    #print('fff', H, W, C)
    f = tf.reshape(features, [H*W, C])
    gram = tf.matmul( tf.transpose(f), f)

    if normalize:
        gram /= tf.cast(H*W, dtype=tf.float32)
    return gram

@tf.function
def style_loss(feats, style_layers, style_targets, style_weights):
    style_loss = 0
    for i in range(len(style_layers)):
        current_style = gram_matrix(feats[style_layers[i]])
        sl = style_weights[i] * tf.reduce_mean(tf.square(current_style - style_targets[i]))
        #print(i, sl)
        style_loss += sl
    return style_loss


@tf.function
def total_variation_loss(img, total_variation_weight):
    return total_variation_weight * tf.image.total_variation(img)



if __name__=='__main__':
    m = SqueezeNet()

    all_n = []
    
    all_n = get_ckpt_weights('../squeezenet_weights/squeezenet.ckpt')
    
    all_weights = []
    for w in m.model.weights:
        all_weights.append(all_n[weight_to_weight[w.name]])

    m.set_weights(all_weights)
    m.trainable = False

    sw = 0.012

    params1 = {
        #'content_image' : 'private/greg.jpg',
        #'content_image' : 'private/kris.jpg',
        #'content_image' : 'private/whats.jpeg',
        
        'content_image' : 'styles/blackpool.jpg',


        #'style_image' : 'styles/composition_vii.jpg',
        #'style_image' : 'styles/muse.jpg',

        #'style_image': 'styles/starry_night.jpg',
        #'style_image': 'styles/impr_sunset.jpg',
        #'style_image': 'styles/the_scream.jpg',
        #'style_image': 'styles/mona.jpg',

        'style_image': 'styles/farm-painting.jpg',        

        #'style_image': 'styles/impr2.jpg',        
        #'style_image': 'styles/dali.jpg',
        #'style_image': 'styles/cubism.jpg',                

        #'style_image': 'styles/liberty.jpeg',

        'content_layers' : [2],
        'content_weight' : 2500, 
        'style_layers' : (2, 3, 5, 6, 9), #, 10),
        'style_weights' : (sw*20000, sw*2000, sw*400, sw*80, sw), #(20, 0,0,0,32), #
        'total_variation_weight' : 0.005,
        'init_random': True,
        'model': m,

        'create_gif' : False,
        'save_images' : False, 
        'show_interval' : 100,
        'max_iter': 400,
        'image_size': 448,
        'lr': 0.1
    }

    style_transfer(**params1)
