import tensorflow as tf
import cv2
from Project2.enums import MASK_STYLE
from config import Config
import numpy as np
from utils import RLE

def load_png_image_and_resize_tf(image_path:str)->tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    tf_png_image = tf.image.decode_png(image_bytes, channels=3, dtype=tf.uint16)
    tf_resized_png_image = tf.image.resize(tf_png_image, tf.constant(Config.IMAGE_SHAPE[0]), tf.constant(Config.IMAGE_SHAPE[1]))
    return tf_resized_png_image

def load_mask_tf(RLE_strings:str, mask_shape:tuple[int, int]):
    masks_tf = [__decode_RLE_to_mask(RLE_string) for RLE_string in RLE_strings]
    if Config.MASK_STYLE == MASK_STYLE.MULTI_CLASS_MULTI_LABEL:
        return __concat_masks_tf(masks_tf)
    elif Config.MASK_STYLE == MASK_STYLE.MULTI_CLASS_ONE_LABEL:
        return __encode_masks_into_one_label(masks_tf)


def __decode_RLE_to_mask(RLE_string, mask_shape):
    decoded = RLE.decode(RLE_string, mask_shape)
    mask = tf.expand_dims(decoded, axis=-1)
    mask_size_tf = tf.constant(Config.SEGMENT_SHAPE[0]), tf.constant(Config.SEGMENT_SHAPE[1])
    resized_mask = tf.image.resize(mask, size=mask_size_tf, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, dtype=tf.uint8)
    return resized_mask

def __concat_masks_tf(masks_tf, axis=-1):
    return tf.concat(masks_tf, axis=axis)

def __encode_masks_into_one_label(masks_tf):
    
    _masks_tf = tf.zeros((*Config.SEGMENT_SHAPE, 1), dtype=tf.uint8)
    mask_index = len(masks_tf)-1

    _masks_tf = masks_tf[mask_index]*tf.constant(mask_index+1, dtype=tf.uint8)
    mask_index -= 1

    for mask_tf in reversed(masks_tf[:-1]):
        _masks_tf = tf.where(mask_tf==tf.constant(1, dtype=tf.uint8), tf.constant(mask_index+1, dtype=tf.uint8), _masks_tf)
        mask_index -= 1
    
    return _masks_tf


def load_gray_16_image(image_path:str, normalize=True)->np.ndarray:
    cv2_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if normalize:
        cv2_image = cv2_image/65535.
    return cv2_image