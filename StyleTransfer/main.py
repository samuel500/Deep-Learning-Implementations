"""
Based on:
https://arxiv.org/pdf/1508.06576.pdf
"""

import tensorflow as tf 

import numpy as np 

from pprint import pprint
from copy import deepcopy
from utils import weight_to_weight, get_ckpt_weights
import matplotlib.pyplot as plt

from image_utils import load_image, preprocess_image, deprocess_image

from squeezenet import SqueezeNet



def style_transfer(content_image, style_image, content_layers, content_weight,
                   style_layers, style_weights, total_variation_weight, model, init_random = False):
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
    """
    content_img = preprocess_image(load_image(content_image))
    feats = model.get_layers(content_img[None])
    content_targets = [feats[i] for i in content_layers]

    # Extract features from the style image
    style_img = preprocess_image(load_image(style_image))
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

    
    lr = 0.2
    max_iter = 200

    lr_var = tf.Variable(lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_var)

    img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess_image(content_img))
    axarr[1].imshow(deprocess_image(style_img))
    plt.show()
    plt.figure()
    


    for t in range(max_iter):

        with tf.GradientTape() as tape:

            tape.watch(img_var)
            feats = model.get_layers(img_var)
            content_feats = [feats[i] for i in content_layers]

            # Compute loss
            c_loss = content_loss(content_weight, content_feats, content_targets)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(img_var, tv_weight)
            #print('content_loss', c_loss)
            #print('style_loss', s_loss)
            #print('tv_loss', t_loss)

            loss =  s_loss + t_loss + c_loss
            print(t, ' Loss:', loss)

        gradients = tape.gradient(loss, [img_var])

        optimizer.apply_gradients(zip(gradients, [img_var]))


        img_var.assign(tf.clip_by_value(img_var, -1.5, 1.5))

        if t % 50 == 0:
            print('Iteration {}'.format(t))
            image = img_var
            plt.imshow(deprocess_image(image[0], rescale=True))
            plt.axis('off')
            plt.show()

    print('Iteration {}'.format(t))
    image = img_var       
    plt.imshow(deprocess_image(image[0], rescale=True))
    plt.axis('off')
    plt.show()


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
def total_variation_loss(img, tv_weight):
    return tv_weight * tf.image.total_variation(img)



if __name__=='__main__':
    m = SqueezeNet()

    all_n = []
    
    all_n = get_ckpt_weights('./model_weights/squeezenet.ckpt')
    
    all_weights = []
    for w in m.model.weights:
        all_weights.append(all_n[weight_to_weight[w.name]])

    m.set_weights(all_weights)
    m.trainable = False

    sw = 0.01

    params1 = {
        #'content_image' : 'styles/greg.jpg',
        'content_image' : 'styles/blackpool.jpg',

        'style_image' : 'styles/composition_vii.jpg',
        #'style_image': 'styles/starry_night.jpg',
        #'style_image': 'styles/impr_sunset.jpg',
        #'style_image': 'styles/the_scream.jpg',
        #'style_image': 'styles/mona.jpg',

        #'style_image': 'styles/farm-painting.jpg',        

        #'style_image': 'styles/impr2.jpg',        
    
        #'style_image': 'styles/liberty.jpeg',

        'content_layers' : [2],
        'content_weight' : 1800, 
        'style_layers' : (2, 3, 5, 6, 9), #, 10),
        'style_weights' : (sw*10000, sw*2000, sw*400, sw*80, sw), #(20, 0,0,0,32), #
        'total_variation_weight' : 0.004,
        'init_random': False,
        'model': m
    }

    style_transfer(**params1)
