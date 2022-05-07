import tensorflow as tf
from typing import Union

def make_sequence_conv2d_bn_act(
    x,
    filters, 
    kernel_size,
    activation='relu',
    strides=(1,1),
    padding='valid',
    kernel_initializer='glorot_uniform'
    ):
    
    conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)
    bn = tf.keras.layers.BatchNormalization()
    act = tf.keras.layers.Activation(activation)

    x = conv2d(x)
    x = bn(x)
    x = act(x)

    return x
    
