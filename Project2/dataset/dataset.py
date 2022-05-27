from calendar import day_abbr
from typing import Tuple

from click import argument
from config import Config
from preprocess.dataframe import transform
from preprocess.dataframe import split
from preprocess.image import loader
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import cv2
from glob import glob
import os
import albumentations as A


def make_dataset(paths, augment=True):
    d1 = tf.data.Dataset.from_tensor_slices(paths[:, 0])
    d2 = tf.data.Dataset.from_tensor_slices(paths[:, 1])
    d3 = tf.data.Dataset.from_tensor_slices(paths[:, 2])
    dataset = tf.data.Dataset.zip((d1,d2,d3))
    dataset = dataset.shuffle(paths.shape[0], reshuffle_each_iteration=True)
    dataset = dataset.map(load_each_paths_to_tensor,
            num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        dataset = dataset.map(data_augmentation_tf, num_parallel_calls=Config.AUTOTUNE)
    dataset = dataset.batch(Config.BATCH_SIZE)
    dataset = dataset.prefetch(Config.AUTOTUNE)
    return dataset

def make_dataset_old():
    paths = get_path_datasets()
    image_path_dataset = tf.data.Dataset.from_tensor_slices(paths[:,0])
    mask_path_dataset = tf.data.Dataset.from_tensor_slices(paths[:,1])
    weight_map_path_dataset = tf.data.Dataset.from_tensor_slices(paths[:,2])
    dataset = tf.data.Dataset.zip((image_path_dataset, weight_map_path_dataset, mask_path_dataset))
    dataset = dataset.shuffle(38000)
    dataset = dataset.map(load_each_paths_to_tensor,
          num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(Config.BATCH_SIZE)
    dataset = dataset.prefetch(Config.AUTOTUNE)
    return dataset

def load_each_paths_to_tensor(image_path, mask_path, weight_map_path):
    x_image = load_16bit_grayscale_png_image_and_resize_tf(image_path)
    x_weight_map = tf.expand_dims(tf.numpy_function(load_npy_and_resize, [weight_map_path], tf.float32), axis=-1)
    y_mask = tf.numpy_function(load_npy_and_resize, [mask_path], tf.float32)
    x_image = __normalize(x_image)
    # x_weight_map = __normalize(x_weight_map)
    return x_image, x_weight_map, y_mask

def data_augmentation_tf(x_image, x_weight_map, y_mask):
    x_image, x_weight_map, y_mask = tf.numpy_function(func=data_augmentation, inp=[x_image, x_weight_map, y_mask], Tout=[tf.float32, tf.float32, tf.float32])
    return (x_image,x_weight_map), y_mask

def data_augmentation(x_image, x_weight_map, y_mask):
    data_transforms = A.Compose(
            [A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=Config.IMAGE_SHAPE[0]//20, max_width=Config.IMAGE_SHAPE[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], 
            additional_targets={'weight_map': 'image', 'mask': 'image'},
            p=0.66
    )
    aug_data = data_transforms(image=x_image, weight_map=x_weight_map, mask=y_mask)
    x_image = aug_data['image']
    x_weight_map = aug_data['weight_map']
    y_mask = aug_data['mask']
    return x_image, x_weight_map, y_mask


def load_npy_and_resize(feature_path):
  feature = np.load(feature_path).astype(np.float32)
  feature = cv2.resize(feature, Config.IMAGE_SHAPE)
  return feature

def make_full_path_from_transformed_df(df, include_weight_map=True, image_format='.png', mask_format='.npy', weight_map_format='.npy'):
    infos = np.stack([df["case_id_str"], df["day_number_str"] ,df['filename']]).T
    image_base = Config.TRAIN_DIR
    mask_base = Config.MASK_DIR
    weight_map_base = Config.WEIGHT_MAP_DIR
    
    full_paths = []
    for info in infos:
        _case, _day, _filename = info
        tree_path = os.path.join(_case, _case+"_"+_day,"scans")
        image_path = os.path.join(image_base, tree_path, _filename+image_format)
        mask_path = os.path.join(mask_base, tree_path, _filename+mask_format)
        if include_weight_map:
            weight_map_path = os.path.join(weight_map_base, tree_path ,_filename+weight_map_format)
            full_paths.append((image_path, mask_path, weight_map_path))
        else:
            full_paths.append((image_path, mask_path))
    return np.array(full_paths)

def get_path_datasets(shuffle=False):

    image_paths = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)
    image_paths = sorted(image_paths)

    mask_paths = glob(os.path.join(Config.MASK_DIR, "*.npy"))
    mask_paths = sorted(mask_paths)

    weight_map_paths = glob(os.path.join(Config.WEIGHT_MAP_DIR, "*.npy"))
    weight_map_paths = sorted(weight_map_paths)

    if not (len(image_paths) == len(mask_paths) == len(weight_map_paths)):
        raise Exception("image, mask, weight map datas not equal")
    
    dataset = np.stack([image_paths,mask_paths,weight_map_paths], axis=1)
    if shuffle:
        np.random.shuffle(dataset)

    return dataset

def load_npy_resize(npy_path):
    npy = np.load(npy_path).astype(np.float32)
    npy = cv2.resize(npy, Config.IMAGE_SHAPE)
    return npy

def load_16bit_grayscale_png_image_and_resize_tf(image_path:str)->tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    png_image = tf.image.decode_png(image_bytes, channels=1, dtype=tf.uint16)
    png_image = tf.cast(png_image, dtype=tf.float32)

    resized_png_image = tf.image.resize(png_image, Config.IMAGE_SHAPE)

    return resized_png_image


def make_train_validation_dataset()->Tuple[tf.data.Dataset, tf.data.Dataset]:

    df_dir = transform.get_train_csv_to_trainable_csv()
    df = pd.read_csv(df_dir)
    transform.make_RLE_encoded_masks_to_single_png_mask_and_save(df)
    transform.make_mask_to_weight_map_and_save()

    train_df, valid_df = split.create_kfold_train_validation_dataframe(df)
    train_dataset, valid_dataset = pass_through_tf_pipeline_from_train_valid_dataframe_to_dataset(train_df, valid_df)
    return train_dataset, valid_dataset

def pass_through_tf_pipeline_from_train_valid_dataframe_to_dataset(train_df:pd.DataFrame, valid_df:pd.DataFrame)->Tuple[tf.data.Dataset, tf.data.Dataset]:
    
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
    data = data / (K.max(data, axis=0) + K.epsilon())
    mask = mask / (K.max(mask, axis=0) + K.epsilon())
    weight_map = weight_map / (K.max(weight_map, axis=0) + K.epsilon())
    weight_map *= 2.
    return data, (mask, weight_map)
    
def __normalize(x):
    x = x - K.min(x, axis=0)
    x = x / (K.max(x, axis=0) + K.epsilon()) 
    return x

def __batch_dataset(dataset:tf.data.Dataset)->tf.data.Dataset:
    dataset = dataset.batch(Config.Dataset.BATCH_SIZE, drop_remainder=True)
    return dataset


def __argument_dataset(dataset):
    return dataset

def __normalize_dataset_batch(dataset:tf.data.Dataset):
    dataset = dataset.map(lambda batch, mask: (loader.normalize_batch_tf(batch), mask), num_parallel_calls=Config.Dataset.AUTOTUNE)
    return dataset