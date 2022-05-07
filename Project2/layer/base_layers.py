import tensorflow as tf
from config import Config

def conv2d_bn_activation(
        x,
        filters, 
        kernel_size,
        activation=Config.Train.activation,
        strides=(1,1),
        padding='valid',
        kernel_initializer=Config.Train.kernel_initializer
    ):

    conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)
    bn = tf.keras.layers.BatchNormalization()
    act = tf.keras.layers.Activation(activation)

    x = conv2d(x)
    x = bn(x)
    x = act(x)

    return x

def concat(inputs, **kwargs):
    return tf.keras.layers.Concatenate(**kwargs)(inputs)

def maxpool2d(x, **kwargs):
    return tf.keras.layers.MaxPool2D(**kwargs)(x)


def upconv2d(x, **kwargs):
    return tf.keras.layers.Conv2DTranspose(**kwargs)(x)






