from multiprocessing import reduction
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
def weighted_loss(y_true, y_pred):
    return 

def calculate_bce_loss(y_true, y_pred, reduction=True):
    loss = y_true * K.log(y_pred)
    loss += (1-y_true) * K.log(1-y_pred)
    if reduction:
        loss = K.mean(loss)
    return -loss

def calculate_bce_weighted_loss(y_true, y_pred, y_weight_map):
    loss = calculate_bce_loss(y_true, y_pred, reduction=False)
    return K.mean(K.sum(loss * y_weight_map, axis=0))


def bce(from_logits=False, label_smoothing=0.0, axis=-1, reduction=keras.losses.Reduction.AUTO):
    return keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, axis=axis, reduction=reduction)
    
def binary_crossentropy(y_true, y_pred, y_weight_map):
    loss_fn = keras.losses.BinaryCrossentropy(reduction=keras.losses.Reduction.NONE)
    loss = loss_fn(y_true, y_pred)
    weighted_loss = y_weight_map * loss
    tf.print(y_true.shape)
    tf.print(y_pred.shape)
    tf.print(y_weight_map.shape)
    tf.print(weighted_loss.shape)
    return weighted_loss

def categorical_crossentropy(y_true, y_pred, y_weight_map):
    loss_fn = keras.losses.CategoricalCrossentropy(reduction=keras.losses.Reduction.NONE)
    loss = loss_fn(y_true, y_pred)
    weighted_loss = y_weight_map * loss
    tf.print(y_true.shape)
    tf.print(y_pred.shape)
    tf.print(y_weight_map.shape)
    tf.print(weighted_loss.shape)
    return weighted_loss
