from cv2 import add
from numpy import pad
import tensorflow as tf
from tensorflow import keras
from layer.base_layers import *

def get_unet_model(from_logits=False):

    x = Config.Train.inputs
    
    x = conv2d_bn_activation(x, 64, 3, padding='same')
    x = conv2d_bn_activation(x, 64, 3, padding='same')

    skip_1 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 128, 3, padding='same')
    x = conv2d_bn_activation(x, 128, 3, padding='same')

    skip_2 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 256, 3, padding='same')
    x = conv2d_bn_activation(x, 256, 3, padding='same')

    skip_3 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 512, 3, padding='same')
    x = conv2d_bn_activation(x, 512, 3, padding='same')

    skip_4 = x
    x = maxpool2d(x, 2, 2)

    x = conv2d_bn_activation(x, 1024, 3, padding='same')
    x = conv2d_bn_activation(x, 1024, 3, padding='same')

    x = upconv2d(x)

    x = concat([skip_4, x])

    x = conv2d_bn_activation(x, 512, 3, padding='same')
    x = conv2d_bn_activation(x, 512, 3, padding='same')

    x = upconv2d(x)

    x = concat([skip_3, x])

    x = conv2d_bn_activation(x, 256, 3, padding='same')
    x = conv2d_bn_activation(x, 256, 3, padding='same')

    x = upconv2d(x)
    x = concat([skip_2, x])

    x = conv2d_bn_activation(x, 128, 3, padding='same')
    x = conv2d_bn_activation(x, 128, 3, padding='same')

    x = upconv2d(x)
    x = concat([skip_1, x])

    x = conv2d_bn_activation(x, 64, 3, padding='same')
    x = conv2d_bn_activation(x, 64, 3, padding='same')
    x = conv2d_bn_activation(x, Config.N_LABELS, 3, padding='same')

    if not from_logits:
        x = activation(x, ACTIVATION.SIGMOID)
        
    outputs = x 

    model = keras.Model(inputs=Config.Train.inputs, outputs=outputs)
    
    return model