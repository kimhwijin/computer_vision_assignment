import tensorflow as tf
import cv2
from enums import MASK_STYLE
from config import Config
import numpy as np
from utils import RLE

def load_16bit_grayscale_png_image_and_resize_tf(image_path:str)->tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    png_image = tf.image.decode_png(image_bytes, channels=1, dtype=tf.uint16)
    png_image = tf.cast(png_image, dtype=tf.float32)

    resized_shape = (tf.constant(Config.IMAGE_SHAPE[0]), tf.constant(Config.IMAGE_SHAPE[1]))
    resized_png_image = tf.image.resize(png_image, resized_shape)

    return resized_png_image

def load_mask_and_resize_tf(id_x:str)->tf.Tensor:
    filepath = tf.strings.join([Config.MASK_DIR, id_x], separator='/')
    image_bytes = tf.io.read_file(filepath)
    png_image = tf.image.decode_png(image_bytes, channels=3)
    png_image = tf.cast(png_image, dtype=tf.float32)

    resized_shape = (tf.constant(Config.IMAGE_SHAPE[0]), tf.constant(Config.IMAGE_SHAPE[1]))
    resized_png_image = tf.image.resize(png_image, resized_shape)
    return resized_png_image

def load_weight_map_and_resize_tf(id_x:str)->tf.Tensor:
    # id_x = tf.strings.join(id_x, '.png')
    filepath = tf.strings.join([Config.WEIGHT_MAP_DIR, id_x], separator='/')
    image_bytes = tf.io.read_file(filepath)
    png_image = tf.image.decode_png(image_bytes, channels=3)
    png_image = tf.cast(png_image, dtype=tf.float32)
    
    resized_shape = (tf.constant(Config.IMAGE_SHAPE[0]), tf.constant(Config.IMAGE_SHAPE[1]))
    resized_png_image = tf.image.resize(png_image, resized_shape)
    return resized_png_image

def load_gray_16_image(image_path:str, normalize:bool=True)->np.ndarray:
    cv2_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if normalize:
        cv2_image = cv2_image/65535.
    return cv2_image

def normalize_batch_tf(batch:tf.Tensor, axis=0)->tf.Tensor:
    _mean = tf.math.reduce_mean(batch, keepdims=True ,axis=axis)
    _var = tf.math.reduce_variance(batch, keepdims=True, axis=axis)
    normalized_batch = (batch - _mean)/(_var+tf.keras.backend.epsilon())
    return normalized_batch
