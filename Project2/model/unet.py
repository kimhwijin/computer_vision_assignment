import tensorflow as tf
from tensorflow import keras
from layer.base_layers import *

def get_unet_model():

    x = Config.Train.inputs

    x = conv2d_bn_activation(x, 64, 3)
    x = conv2d_bn_activation(x, 64, 3)

    skip_1 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 128, 3)
    x = conv2d_bn_activation(x, 128, 3)

    skip_2 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 256, 3)
    x = conv2d_bn_activation(x, 256, 3)

    skip_3 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 512, 3)
    x = conv2d_bn_activation(x, 512, 3)

    skip_4 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 1024, 3)
    x = conv2d_bn_activation(x, 1024, 3)

    outputs = x 
    model = keras.Model(inputs=Config.Train.inputs, outputs=outputs)
    
    return x 