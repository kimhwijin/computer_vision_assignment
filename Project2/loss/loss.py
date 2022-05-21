from multiprocessing import reduction
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf

def weighted_loss(y_true, y_pred):
    return 

def binary_crossentropy(y_true, y_pred):
    return 


def categorical_crossentropy(y_true, y_pred, **kwargs):
    return K.mean(K.sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1))

def generalized_dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    dice = (2. * intersection + K.epsilon()) / (union + K.epsilon())
    return dice

def dice_loss(y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss

def focal_loss_with_logits(logits, targets, y_pred, alpha=0.25, gamma=2.):
    weight_a = alpha * (1-y_pred) ** gamma * targets
    weight_b = (1-alpha) * y_pred ** gamma * (1-targets)
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

def focal_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
    logits = tf.math.log(y_pred / (1-y_pred))
    loss = focal_loss_with_logits(logits, y_true, y_pred)
    return tf.reduce_mean(loss)

def tversky_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1-y_pred_f))
    false_pos = K.sum((1-y_true_f) * y_pred_f)
    alpha = 0.7
    smooth = 1.
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1-alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1. - tversky_index(y_true, y_pred)

def focal_tversky(y_true, y_pred):
    pt_1 = tversky_index(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)
