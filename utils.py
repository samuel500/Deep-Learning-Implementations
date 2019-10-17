

from tensorflow.python import pywrap_tensorflow


def get_ckpt_weights(file_name):
    """
    returns dictionary of numpy arrays of all weights in a checkpoint file
    """
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_to_dtype_map = reader.get_variable_to_dtype_map()
    to_ret = {}
    for key, value in sorted(var_to_shape_map.items()):

        to_ret[key] = reader.get_tensor(key)
    return to_ret




weight_to_weight = {
        'conv2d/kernel:0': 'features/layer0/weights',
        'conv2d/bias:0': 'features/layer0/bias',
        'fire_layer/conv2d_1/kernel:0': 'features/layer3/fire/squeeze/weights',
        'fire_layer/conv2d_1/bias:0': 'features/layer3/fire/squeeze/bias',
        'fire_layer/conv2d_2/kernel:0': 'features/layer3/fire/e11/weights',
        'fire_layer/conv2d_2/bias:0': 'features/layer3/fire/e11/bias',
        'fire_layer/conv2d_3/kernel:0': 'features/layer3/fire/e33/weights',
        'fire_layer/conv2d_3/bias:0': 'features/layer3/fire/e33/bias',

        'fire_layer_1/conv2d_4/kernel:0': 'features/layer4/fire/squeeze/weights',
        'fire_layer_1/conv2d_4/bias:0': 'features/layer4/fire/squeeze/bias',
        'fire_layer_1/conv2d_5/kernel:0': 'features/layer4/fire/e11/weights',
        'fire_layer_1/conv2d_5/bias:0':'features/layer4/fire/e11/bias',
        'fire_layer_1/conv2d_6/kernel:0': 'features/layer4/fire/e33/weights',
        'fire_layer_1/conv2d_6/bias:0': 'features/layer4/fire/e33/bias',

        'fire_layer_2/conv2d_7/kernel:0': 'features/layer6/fire/squeeze/weights',
        'fire_layer_2/conv2d_7/bias:0': 'features/layer6/fire/squeeze/bias',
        'fire_layer_2/conv2d_8/kernel:0': 'features/layer6/fire/e11/weights',
        'fire_layer_2/conv2d_8/bias:0': 'features/layer6/fire/e11/bias',
        'fire_layer_2/conv2d_9/kernel:0': 'features/layer6/fire/e33/weights',
        'fire_layer_2/conv2d_9/bias:0': 'features/layer6/fire/e33/bias',

        'fire_layer_3/conv2d_10/kernel:0': 'features/layer7/fire/squeeze/weights',
        'fire_layer_3/conv2d_10/bias:0': 'features/layer7/fire/squeeze/bias',
        'fire_layer_3/conv2d_11/kernel:0': 'features/layer7/fire/e11/weights',
        'fire_layer_3/conv2d_11/bias:0': 'features/layer7/fire/e11/bias',
        'fire_layer_3/conv2d_12/kernel:0': 'features/layer7/fire/e33/weights',
        'fire_layer_3/conv2d_12/bias:0': 'features/layer7/fire/e33/bias',

        'fire_layer_4/conv2d_13/kernel:0': 'features/layer9/fire/squeeze/weights',
        'fire_layer_4/conv2d_13/bias:0': 'features/layer9/fire/squeeze/bias',
        'fire_layer_4/conv2d_14/kernel:0': 'features/layer9/fire/e11/weights',
        'fire_layer_4/conv2d_14/bias:0': 'features/layer9/fire/e11/bias',
        'fire_layer_4/conv2d_15/kernel:0': 'features/layer9/fire/e33/weights',
        'fire_layer_4/conv2d_15/bias:0': 'features/layer9/fire/e33/bias',

        'fire_layer_5/conv2d_16/kernel:0': 'features/layer10/fire/squeeze/weights',
        'fire_layer_5/conv2d_16/bias:0': 'features/layer10/fire/squeeze/bias',
        'fire_layer_5/conv2d_17/kernel:0': 'features/layer10/fire/e11/weights',
        'fire_layer_5/conv2d_17/bias:0': 'features/layer10/fire/e11/bias',
        'fire_layer_5/conv2d_18/kernel:0': 'features/layer10/fire/e33/weights',
        'fire_layer_5/conv2d_18/bias:0': 'features/layer10/fire/e33/bias',

        'fire_layer_6/conv2d_19/kernel:0': 'features/layer11/fire/squeeze/weights',
        'fire_layer_6/conv2d_19/bias:0': 'features/layer11/fire/squeeze/bias',
        'fire_layer_6/conv2d_20/kernel:0': 'features/layer11/fire/e11/weights',
        'fire_layer_6/conv2d_20/bias:0': 'features/layer11/fire/e11/bias',
        'fire_layer_6/conv2d_21/kernel:0': 'features/layer11/fire/e33/weights',
        'fire_layer_6/conv2d_21/bias:0': 'features/layer11/fire/e33/bias',

        'fire_layer_7/conv2d_22/kernel:0': 'features/layer12/fire/squeeze/weights',
        'fire_layer_7/conv2d_22/bias:0': 'features/layer12/fire/squeeze/bias',
        'fire_layer_7/conv2d_23/kernel:0': 'features/layer12/fire/e11/weights',
        'fire_layer_7/conv2d_23/bias:0': 'features/layer12/fire/e11/bias',
        'fire_layer_7/conv2d_24/kernel:0': 'features/layer12/fire/e33/weights',
        'fire_layer_7/conv2d_24/bias:0': 'features/layer12/fire/e33/bias',

        'conv2d_25/kernel:0': 'classifier/layer1/weights',
        'conv2d_25/bias:0': 'classifier/layer1/bias'
}
