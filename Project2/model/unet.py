from cv2 import add
from numpy import pad
from scipy.fft import skip_backend
import tensorflow as tf
from tensorflow import keras
from layer.base_layers import *
import tensorflow.keras.backend as K

def get_weighted_unet_model():
    image_input = tf.keras.layers.Input(shape=Config.MODEL_IMAGE_INPUT_SHAPE)

    x, skip_connection = contracting_path(image_input)
    image_output = expending_path(x, skip_connection)

    weight_map_input = keras.Input(Config.MODEL_WEIGHT_MAP_INPUT_SHAPE)
    outputs = applying_eighted_map(weight_map_input, image_output)

    model = keras.Model(inputs=[image_input, weight_map_input], outputs=[outputs])
    return model


def get_unet_model(from_logits=False):
    inputs = tf.keras.layers.Input((256,256,1))
    x = inputs
    
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
        x = activation(x, ACTIVATION.SIGMOID.value)
        
    outputs = x 

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def contracting_path(image_input):
    skip_connection = []
    x = image_input

    x = keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    skip_connection.append(x)
    x = keras.layers.MaxPool2D(2)(x)

    x = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    skip_connection.append(x)
    x = keras.layers.MaxPool2D(2)(x)

    x = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    skip_connection.append(x)
    x = keras.layers.MaxPool2D(2)(x)

    x = keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    skip_connection.append(x)
    x = keras.layers.MaxPool2D(2)(x)

    x = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(1024, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    return x, skip_connection

def expending_path(x, skip_connection):

    x = keras.layers.Conv2DTranspose(512, 3, 2, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Concatenate()([x, skip_connection.pop()])

    x = keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(512, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    x = keras.layers.Conv2DTranspose(256, 3, 2, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Concatenate()([x, skip_connection.pop()])
    
    x = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(256, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    x = keras.layers.Conv2DTranspose(128, 3, 2, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Concatenate()([x, skip_connection.pop()])

    x = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(128, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    x = keras.layers.Conv2DTranspose(64, 3, 2, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Concatenate()([x, skip_connection.pop()])

    x = keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)
    x = keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=Config.KERNEL_INITIALIZER, activation='relu')(x)

    x = keras.layers.Conv2D(Config.N_LABELS, 1, kernel_initializer=Config.KERNEL_INITIALIZER, activation='softmax')(x)
    image_output = x
    return image_output


def applying_eighted_map(weight_map_input, image_output):

    normalized_activation = keras.layers.Lambda(lambda x : x / tf.reduce_sum(x, axis=-1, keepdims=True))(image_output)
    _epsilon = K.epsilon()
    clip_activation = keras.layers.Lambda(lambda x : tf.clip_by_value(x, _epsilon, 1 - _epsilon))(normalized_activation)
    log_activation = keras.layers.Lambda(lambda x : K.log(x))(clip_activation)
    weighted_softmax = keras.layers.multiply([log_activation, weight_map_input])
    outputs = weighted_softmax

    return outputs
