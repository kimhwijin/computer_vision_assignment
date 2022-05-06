from config import Config
from preprocess.dataframe.transform import load_and_preprocess_train_dataframe
from preprocess.dataframe.split import create_kfold_train_validation_dataframe
from preprocess.image.loader import load_mask_tf, load_png_image_and_resize_tf, normalize_batch
import pandas as pd
import tensorflow as tf

def make_train_validation_dataset()->tuple[tf.data.Dataset, tf.data.Dataset]:
    df = load_and_preprocess_train_dataframe()
    train_df, valid_df = create_kfold_train_validation_dataframe(df)
    train_dataset, valid_dataset = make_train_valid_dataset(train_df, valid_df)
    return train_dataset, valid_dataset


def make_train_valid_dataset(train_df:pd.DataFrame, valid_df:pd.DataFrame)->tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dataset = __make_dataset_from_filepath_segmentation_shape(train_df)
    valid_dataset = __make_dataset_from_filepath_segmentation_shape(valid_df)

    train_dataset = __batch_dataset(train_dataset)
    valid_dataset = __batch_dataset(valid_dataset)

    train_dataset = __shuffle_dataset(train_dataset, Config.Dataset.TRAIN_SHUFFLE_BUFFER_SIZE)
    valid_dataset = __shuffle_dataset(valid_dataset, Config.Dataset.VALIDATION_SHUFFLE_BUFFER_SIZE)

    train_dataset = __transform_filepath_to_image_dataset(train_dataset)
    valid_dataset = __transform_filepath_to_image_dataset(valid_dataset)
    # dataset = __argument_dataset(dataset)
    # dataset = __argument_dataset(dataset)

    train_dataset = __normalize_dataset_batch(train_dataset)
    valid_dataset = __normalize_dataset_batch(valid_dataset)

    train_dataset.prefetch(Config.Dataset.AUTOTUNE)
    valid_dataset.prefetch(Config.Dataset.AUTOTUNE)

    return train_dataset, valid_dataset


def __make_dataset_from_filepath_segmentation_shape(df:pd.DataFrame)->tf.data.Dataset:
    slices = (
        df["full_filepath"],
        (df["large_bowel_RLE_encoded"], df["small_bowel_RLE_encoded"], df["large_bowel_RLE_encoded"]),
        (df["slice_width"], df["slice_height"])
    )
    dataset = tf.data.Dataset.from_tensor_slices(slices)    
    return dataset

def __batch_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.batch(Config.Dataset.BATCH_SIZE, drop_remainder=True)
    return dataset

def __transform_filepath_to_image_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.map(
        lambda full_filepath, RLE_strings, mask_shape:
        (load_png_image_and_resize_tf(full_filepath),load_mask_tf(RLE_strings, mask_shape)),
        num_parallel_calls=Config.Dataset.AUTOTUNE
    )
    return dataset

def __shuffle_dataset(dataset:tf.data.Dataset, buffer_size:int)->tf.data.Dataset:
    dataset = dataset.shuffle(buffer_size)
    return dataset

def __argument_dataset(dataset):
    return dataset

def __normalize_dataset_batch(dataset:tf.data.Dataset):
    dataset = dataset.map(lambda batch, mask: (normalize_batch(batch), mask), num_parallel_calls=Config.Dataset.AUTOTUNE)
    return dataset

