from tensorflow import keras
import tensorflow.keras.backend as K
def weighted_loss(y_true, y_pred):
    return 

def calculate_bce_loss(y_true, y_pred, reduction=True):
    loss = y_true * K.log(y_pred)
    loss += (1-y_true) * K.log(1-y_pred)
    if reduction:
        loss = K.mean(loss)
    return -loss

def calculate_bce_weighted_loss(y, y_pred):
    y_true, y_weighted = y
    loss = calculate_bce_loss(y_true, y_pred, reduction=False)
    return K.mean(loss * y_weighted)


def bce(from_logits=False, label_smoothing=0.0, axis=-1, reduction=keras.losses.Reduction.AUTO):
    return keras.losses.BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing, axis=axis, reduction=reduction)
    
