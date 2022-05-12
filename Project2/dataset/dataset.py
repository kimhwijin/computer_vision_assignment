from typing import Tuple
from config import Config
from preprocess.dataframe import transform
from preprocess.dataframe import split
from preprocess.image import loader
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

def make_train_validation_dataset()->Tuple[tf.data.Dataset, tf.data.Dataset]:

    df_dir = transform.get_train_csv_to_trainable_csv()
    df = pd.read_csv(df_dir)
    transform.make_RLE_encoded_masks_to_single_png_mask_and_save(df)
    transform.make_mask_to_weight_map_and_save()

    train_df, valid_df = split.create_kfold_train_validation_dataframe(df)
    train_dataset, valid_dataset = pass_through_tf_pipeline_from_train_valid_dataframe_to_dataset(train_df, valid_df)
    return train_dataset, valid_dataset

def pass_through_tf_pipeline_from_train_valid_dataframe_to_dataset(train_df:pd.DataFrame, valid_df:pd.DataFrame)->tuple[tf.data.Dataset, tf.data.Dataset]:
    
    train_dataset = __make_dataset_from_filepath(train_df)
    valid_dataset = __make_dataset_from_filepath(valid_df)
    
    train_dataset = __shuffle_dataset(train_dataset, Config.Dataset.TRAIN_SHUFFLE_BUFFER_SIZE)
    valid_dataset = __shuffle_dataset(valid_dataset, Config.Dataset.VALIDATION_SHUFFLE_BUFFER_SIZE)

    train_dataset = __transform_filepath_to_image_dataset(train_dataset)
    valid_dataset = __transform_filepath_to_image_dataset(valid_dataset)
    
    train_dataset = __batch_dataset(train_dataset)
    valid_dataset = __batch_dataset(valid_dataset)
    
    train_dataset = __argument_dataset(train_dataset)
    valid_dataset = __argument_dataset(valid_dataset)
    
    # train_dataset = __normalize_dataset_batch(train_dataset)
    # valid_dataset = __normalize_dataset_batch(valid_dataset)

    train_dataset = train_dataset.prefetch(Config.Dataset.AUTOTUNE)
    train_dataset = valid_dataset.prefetch(Config.Dataset.AUTOTUNE)

    return train_dataset, valid_dataset


def __make_dataset_from_filepath(df:pd.DataFrame)->tf.data.Dataset:
    slices = (
        df["full_filepath"],
        df["id"].apply(lambda x: x+'.png')
    )
    unziped_datasets = []
    for _slice in slices:
        unziped_datasets.append(tf.data.Dataset.from_tensor_slices(list(_slice)))
    dataset = tf.data.Dataset.zip(tuple(unziped_datasets))
    return dataset

def __shuffle_dataset(dataset:tf.data.Dataset, buffer_size:int)->tf.data.Dataset:
    dataset = dataset.shuffle(buffer_size)
    return dataset

def __transform_filepath_to_image_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.map(__load_resized_png_image, num_parallel_calls=Config.Dataset.AUTOTUNE)
    dataset = dataset.map(__load_resized_mask_weight_map_image, num_parallel_calls=Config.Dataset.AUTOTUNE)
    dataset = dataset.map(__rescale_from_0_upto_1, num_parallel_calls=Config.Dataset.AUTOTUNE)
    return dataset

def __load_resized_png_image(full_filepath, file_id):
    return loader.load_16bit_grayscale_png_image_and_resize_tf(full_filepath), file_id

def __load_resized_mask_weight_map_image(image, file_id):
    return image, (loader.load_mask_and_resize_tf(file_id) , loader.load_weight_map_and_resize_tf(file_id))

def __rescale_from_0_upto_1(x, y):
    data = x
    mask, weight_map = y
    data = data / (K.max(data) + K.epsilon())
    mask = mask / (K.max(mask) + K.epsilon())
    weight_map = weight_map / (K.max(weight_map) + K.epsilon())
    return data, (mask, weight_map)
    

def __batch_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.batch(Config.Dataset.BATCH_SIZE, drop_remainder=True)
    return dataset


def __argument_dataset(dataset):
    return dataset

def __normalize_dataset_batch(dataset:tf.data.Dataset):
    dataset = dataset.map(lambda batch, mask: (loader.normalize_batch_tf(batch), mask), num_parallel_calls=Config.Dataset.AUTOTUNE)
    return dataset

