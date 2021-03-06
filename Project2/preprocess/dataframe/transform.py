import os
from config import Config
import pandas as pd
from glob import glob
from preprocess.image.loader import RLE 
import cv2
import numpy as np

if False:
    df_dir = get_train_csv_to_trainable_csv()
    df = pd.read_csv(df_dir)
    make_RLE_encoded_masks_to_single_png_mask_and_save(df)
    make_weight_map_and_save()


def get_train_csv_and_transform():

    df = pd.read_csv(Config.TRAIN_CSV)
    all_train_images = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)

    df = __split_case_id_and_add_column(df)
    df = __split_day_number_and_add_column(df)
    df = __split_slice_id_and_add_column(df)
    df = __merge_dataframe_with_filepath(df, all_train_images)
    df = __slice_file_height_and_weight(df)
    df = __slice_pixel_spacing(df)
    df = __merge_LF_rows_to_single_row_and_multiple_columns(df)
    df = __reorder_columns_of_dataframe(df, __get_order_of_columns(False))
 
    # df.to_csv(Config.TRAINABLE_CSV)
    return df

def make_RLE_encoded_masks_to_single_png_mask_and_save(df):
    if Config.ALREADY_SAVED_MASK:
        return
    large_bowel_RLE = list(df['large_bowel_RLE_encoded'].fillna(""))
    small_bowel_RLE = list(df['small_bowel_RLE_encoded'].fillna(""))
    stomach_RLE = list(df['stomach_RLE_encoded'].fillna(""))
    RLE_masks = list(zip(large_bowel_RLE, small_bowel_RLE, stomach_RLE))

    seg_height = list(df['slice_height'])
    seg_width = list(df['slice_width'])
    seg_shape = list(zip(seg_width, seg_height))

    file_id = list(df['id'])

    unsaved = __make_RLE_encodes_to_png_image_and_save(RLE_masks, seg_shape, file_id)
    print("unsaved mask : ", unsaved)

def make_mask_to_weight_map_and_save():
    unsaved = 0
    if Config.ALREADY_SAVED_WEIGHT_MAP:
        return unsaved
    mask_paths = glob(Config.MASK_DIR + '/*.png')
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        if np.max(mask) < 0.5:
            weight_map = np.zeros_like(mask)
        else:
            shrink_mask = cv2.erode(mask, np.ones((3,3)))
            weight_map = mask - shrink_mask
        if not cv2.imwrite(Config.WEIGHT_MAP_DIR + '/' + mask_path.split('/')[-1], weight_map):
            unsaved += 1
    print("unsaved weight map : ", unsaved)

def __make_RLE_encodes_to_png_image_and_save(encodeds, shapes, file_ids)->list:
    unsaved = 0
    for encoded, shape, file_id in zip(encodeds, shapes, file_ids):
        masks = []
        for RLE_mask in encoded:
            if RLE_mask != "":
                masks.append(RLE.decode(RLE_mask, shape))
            else:
                masks.append(np.zeros(shape))
        png_image = np.stack(masks, axis=-1)
        filename = os.path.join(Config.MASK_DIR, file_id+'.png')
        
        if not cv2.imwrite(filename, png_image):
            unsaved += 1
    return unsaved

def __split_case_id_and_add_column(df : pd.DataFrame) -> pd.DataFrame:
    df["case_id_str"] = df["id"].apply(lambda x : x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(lambda x : int(x.split("_",2)[0].replace("case", "")))
    return df

def __split_day_number_and_add_column(df:pd.DataFrame) -> pd.DataFrame:
    df["day_number_str"] = df["id"].apply(lambda x : x.split("_", 2)[1])
    df["day_number"] = df["id"].apply(lambda x : int(x.split("_", 2)[1].replace("day", "")))
    return df

def __split_slice_id_and_add_column(df:pd.DataFrame) -> pd.DataFrame:
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])
    return df

def __merge_dataframe_with_filepath(df:pd.DataFrame, globbed_full_filepaths:list) ->pd.DataFrame:
    df["_partial_identifier"] = (globbed_full_filepaths[0].rsplit("/", 4)[0] + "/" +
                                    df["case_id_str"] + "/" + 
                                    df["case_id_str"] + "_" + df["day_number_str"] + "/"
                                    "scans/" +
                                    df["slice_id"]
                                )
    
    _merging_df = pd.DataFrame({
        "_partial_identifier" : [x.rsplit("_", 4)[0] for x in globbed_full_filepaths],
        "full_filepath": globbed_full_filepaths
    })
    df = df.merge(_merging_df, on="_partial_identifier").drop(columns="_partial_identifier")
    return df

def __slice_file_height_and_weight(df:pd.DataFrame)->pd.DataFrame:
    df["slice_height"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[1]))
    df["slice_width"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[2]))
    return df

def __slice_pixel_spacing(df:pd.DataFrame)->pd.DataFrame:
    df["pixel_spacing_height"] = df["full_filepath"].apply(lambda x : float(x[:-4].rsplit("_", 4)[3]))
    df["pixel_spacing_width"] = df["full_filepath"].apply(lambda x : float(x[:-4].rsplit("_", 4)[4]))
    return df

def __merge_LF_rows_to_single_row_and_multiple_columns(df:pd.DataFrame)->pd.DataFrame:
    large_bowel_df = df[df["class"] == "large_bowel"][['id', 'segmentation']].rename(columns={
        "segmentation": "large_bowel_RLE_encoded"
    })
    small_bowel_df = df[df["class"] == "small_bowel"][["id", "segmentation"]].rename(columns= {
        "segmentation": "small_bowel_RLE_encoded"
    })
    stomach_df = df[df["class"] == "stomach"][["id", "segmentation"]].rename(columns={
        "segmentation": "stomach_RLE_encoded"
    })
    df = df.merge(large_bowel_df, on="id", how="left")
    df = df.merge(small_bowel_df, on="id", how="left")
    df = df.merge(stomach_df, on="id", how="left")
    df = df.drop_duplicates(subset=["id", ]).reset_index(drop=True)

    df["large_bowel_flag"] = df["large_bowel_RLE_encoded"].apply(lambda x : not pd.isna(x))
    df["small_bowel_flag"] = df["small_bowel_RLE_encoded"].apply(lambda x : not pd.isna(x))
    df["stomach_flag"] = df["stomach_RLE_encoded"].apply(lambda x : not pd.isna(x))
    df["n_segs"] = df["large_bowel_flag"].astype(int) + df["small_bowel_flag"].astype(int) + df["stomach_flag"].astype(int)
    
    return df

def __reorder_columns_of_dataframe(df:pd.DataFrame, order_of_column:list)->pd.DataFrame:
    order_of_column = [_column for _column in order_of_column if _column in df.columns]
    df = df[order_of_column]
    return df
    
def __get_order_of_columns(is_test:bool)->list:
    order_of_column = ["id", "full_filepath", "n_segs",
                     "large_bowel_RLE_encoded", "large_bowel_flag",
                     "small_bowel_RLE_encoded", "small_bowel_flag",
                     "stomach_RLE_encoded", "stomach_flag",
                     "slice_height", "slice_width", 
                     "pixel_spacing_height", "pixel_spacing_height", 
                     "case_id_str", "case_id", 
                     "day_number_str", "day_number", 
                     "slice_id", "predicted"]
    if is_test:
        order_of_column.insert(1, "class")
    return order_of_column

    

