import sys
sys.path.append("..") # to import things one level up

from squeezenet import SqueezeNet
import tensorflow as tf
import numpy as np

from utils import weight_to_weight, get_ckpt_weights

# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    #return tf.cast(img, tf.uint8)
    return np.array(img).astype(np.uint8)


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


def get_keras_model(names, model_class, model_i=0, show_summary=False):
    #base_model = iv3.InceptionV3(include_top=True, weights='imagenet')

    print(dir(model_class))
    base_model = getattr(model_class, dir(model_class)[model_i])(include_top=True, weights='imagenet')
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