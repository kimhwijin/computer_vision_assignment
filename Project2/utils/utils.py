import numpy as np
import tensorflow as tf
import cv2

class RLE:
    def decode(encoded: str, shape: list, color=1)->np.ndarray:

        np_encoded = np.array(encoded.split(), dtype=int)
        pixel_starts = np_encoded[0::2] - 1
        pixel_lengths = np_encoded[1::2]
        pixel_ends = pixel_starts + pixel_lengths

        if len(shape) == 3:
            h, w, d = shape
            decoded = np.zeros((h*w, d), dtype=np.float32)
        else:
            h, w = shape
            decoded = np.zeros((h*w), dtype=np.float32)
    
        for low, high in zip(pixel_starts, pixel_ends):
            decoded[low:high] = color
        return decoded.reshape(shape)
    
    def decode_tf(encoded, shape):
        shape = tf.convert_to_tensor(shape, tf.int64)
        size = tf.math.reduce_prod(size)

        encoded = tf.strings.split(encoded)
        encoded = tf.strings.to_number(encoded, tf.int64)

        pixel_starts = encoded[::2] - 1
        pixel_lengths = encoded[1::2]

        total_ones = tf.reduce_sum(pixel_lengths)
        ones = tf.ones([total_ones], tf.unit8)

        r = tf.range(total_ones)
        cumulative_length_sum = tf.math.cumsum(pixel_lengths)
        s = tf.searchsorted(cumulative_length_sum, r, 'right')
        
        padded_cumsum = tf.pad(cumulative_length_sum[:-1], [(1,0)])
        idx = r + tf.gater(pixel_starts - padded_cumsum, s)

        decoded_flat_mask = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])

        decoded_mask = tf.reshape(decoded_flat_mask, shape)

        return decoded_mask


    def encode(decoded:np.ndarray)-> str:
        pixels = decoded.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
    
def load_png_image_tf(image_path:str)->tf.Tensor:
    tf_image = tf.io.read_file(image_path)
    tf_png_image = tf.image.decode_png(tf_image, channels=3)
    return tf_png_image

def load_gray_16_image(image_path:str, normalize=True):
    cv2_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    if normalize:
        cv2_image = cv2_image/65535.
    return cv2_image


        
