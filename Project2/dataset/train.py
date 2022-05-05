from config import Config
from preprocess.dataframe.transform import load_and_preprocess_train_dataframe
from preprocess.image.loader import load_mask_tf, load_png_image_and_resize_tf
import pandas as pd
import tensorflow as tf

def make()->pd.DataFrame:
    df = load_and_preprocess_train_dataframe()
    dataset = __make_dataset_from_filepath_segmentation_shape(df)
    dataset = __transform_filepath_to_image_dataset(dataset)
    dataset = __shuffle_dataset(dataset)
    

def __make_dataset_from_filepath_segmentation_shape(df:pd.DataFrame)->tf.data.Dataset:
    slices = (
        df["full_filepath"],
        (df["large_bowel_RLE_encoded"], df["small_bowel_RLE_encoded"], df["large_bowel_RLE_encoded"]),
        (df["slice_width"], df["slice_height"])
    )
    dataset = tf.data.Dataset.from_tensor_slices(slices)    
    return dataset

def __transform_filepath_to_image_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.map(
        lambda full_filepath, RLE_strings, mask_shape:
        (load_png_image_and_resize_tf(full_filepath),load_mask_tf(RLE_strings, mask_shape)),
        num_parallel_calls=Config.Dataset.AUTOTUNE
    )
    return dataset

def __shuffle_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.shuffle(Config.Dataset.SHUFFLE_BUFFER_SIZE)
    return dataset

def __argument_dataset(dataset):
    return dataset

