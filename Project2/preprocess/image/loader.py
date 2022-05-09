import tensorflow as tf
import cv2
from enums import MASK_STYLE
from config import Config
import numpy as np
from utils import RLE

def load_png_image_and_resize_tf(image_path:str)->tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    png_image = tf.image.decode_png(image_bytes, channels=3)

    resized_shape = (tf.constant(Config.IMAGE_SHAPE[0]), tf.constant(Config.IMAGE_SHAPE[1]))
    resized_png_image = tf.image.resize(png_image, resized_shape)

    return resized_png_image

def load_mask(s1,s2,s3, mask_shape):
    t1 = __decode_RLE_to_mask(s1, mask_shape)
    t2 = __decode_RLE_to_mask(s2, mask_shape)
    t3 = __decode_RLE_to_mask(s3, mask_shape)
    if Config.MASK_STYLE == MASK_STYLE.MULTI_CLASS_MULTI_LABEL:
        return tf.concat((t1,t2,t3), axis=-1)

def load_gray_16_image(image_path:str, normalize:bool=True)->np.ndarray:
    cv2_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if normalize:
        cv2_image = cv2_image/65535.
    return cv2_image

def normalize_batch(batch:tf.Tensor, axis=0)->tf.Tensor:
    _mean = tf.math.reduce_mean(batch, keepdims=True ,axis=axis)
    _var = tf.math.reduce_variance(batch, keepdims=True, axis=axis)
    normalized_batch = (batch - _mean)/(_var+tf.keras.backend.epsilon())
    return normalized_batch
    
def __decode_RLE_to_mask(RLE_string, mask_shape)->tf.Tensor:
    if tf.strings.length(RLE_string) == tf.constant(0):
        mask = tf.zeros(mask_shape, dtype=tf.uint8)
    else:
        mask =  RLE.decode_tf(RLE_string, mask_shape)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, size=Config.SEGMENT_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.uint8)
    return mask

def __concat_masks_tf(masks_tf:tf.Tensor, axis=-1)->tf.Tensor:
    return tf.concat(masks_tf, axis=axis)