import pandas as pd
from utils.utils import RLE
import numpy as np
from config import Config
from enums import MASK_STYLE
import cv2
import os

def make_RLE_to_mask_and_save_for_all(df:pd.DataFrame)->pd.DataFrame:
    df[f"{Config.MASK_STYLE.name}_mask_path"] = df.prograss_apply(lambda _row: make_RLE_to_mask_and_save_for_each_row(_row), axis=1)
    return df

def make_RLE_to_mask_and_save_for_each_row(row:pd.DataFrame)->str:
    mask = make_RLE_to_mask_for_each_row(row)
    mask_path = save_mask_if_not_exists(mask, row["id"])
    return mask_path

def make_RLE_to_mask_for_each_row(row:pd.DataFrame)->np.ndarray:
    class_masks = __get_each_class_mask(row)
    mask = __transform_masks_for_style(class_masks)
    mask = __resize(mask, Config.SEGMENT_SHAPE)
    return mask

def save_mask_if_not_exists(mask:np.ndarray, id:str)->str:
    mask_filename = f"{id}_mask"
    mask_path = os.path.join(Config.MASK_DIR, mask_filename)
    if not os.path.exists(mask_path + '.npy'):
        np.save(mask_path, mask)
    return mask_path + '.npy'

def __get_each_class_mask(row:pd.DataFrame)-> tuple[np.ndarray, np.ndarray, np.ndarray]:

    slice_shape = (row["slice_height"], row["slice_width"])

    if not pd.isna(row["large_bowel_RLE_encoded"]):
        large_bowel_mask = RLE.decode(row["large_bowel_RLE_encoded"], slice_shape)
    else:
        large_bowel_mask = np.zeros(slice_shape)
    
    if not pd.isna(row["small_bowel_RLE_encoded"]):
        small_bowel_mask = RLE.decode(row["small_bowel_RLE_encoded"], slice_shape)
    else:
        small_bowel_mask = np.zeros(slice_shape)

    if not pd.isna(row["stomach_RLE_encoded"]):
        stomach_mask = RLE.decode(row["stomach_RLE_encoded"], slice_shape)
    else:
        stomach_mask = np.zeros(slice_shape)

    return large_bowel_mask, small_bowel_mask, stomach_mask

def __transform_masks_for_style(class_masks:tuple)->np.ndarray:
    if Config.MASK_STYLE == MASK_STYLE.MULTI_CLASS_ONE_LABEL:
        mask = __transform_masks_one_label(class_masks)
    elif Config.MASK_STYLE == MASK_STYLE.MULTI_CLASS_MULTI_LABEL:
        mask = __transform_masks_multi_label(class_masks)
    return mask

def __transform_masks_one_label(class_masks)->np.ndarray:
    large_bowel_mask, small_bowel_mask, stomach_mask = class_masks
    mask = stomach_mask * 3
    mask = np.where(small_bowel_mask==1, 2, mask)
    mask = np.where(large_bowel_mask==1, 1, mask)
    return mask

def __transform_masks_multi_label(class_masks)->np.ndarray:
    large_bowel_mask, small_bowel_mask, stomach_mask = class_masks
    mask = np.stack([large_bowel_mask, small_bowel_mask, stomach_mask], axis=-1)
    return mask


def __resize(mask:np.ndarray, shape:tuple[int, int])->np.ndarray:
    mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return mask

