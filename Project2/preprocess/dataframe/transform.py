import os
from config import Config
import pandas as pd
from glob import glob
from preprocess.image.loader import RLE 
import cv2
import numpy as np

def load_row_train_csv_to_dataframe_and_transform_to_usable_dataframe_and_save():
    if not Config.Train.ALREADY_SAVED_SEGMENT:
        df = pd.read_csv(Config.TRAIN_CSV)
        all_train_images = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)

        df = __split_case_id_and_add_column(df)
        df = __split_day_number_and_add_column(df)
        df = __split_slice_id_and_add_column(df)
        df = __merge_dataframe_with_filepath(df, all_train_images)
        df = __slice_file_height_and_weight(df)
        df = __slice_pixel_spacing(df)
        df = __merge_LF_rows_to_single_row_and_multiple_columns(df)
        df = __add_each_weighted_map(df)
        df.to_csv(Config.TRAINABLE_CSV)
    
    return Config.TRAINABLE_CSV

def save_png_images_from_transformed_from_each_RLE_mask_and_weight_map(df):
    if Config.Train.ALREADY_SAVED_SEGMENT:
        return

    large_bowel_RLE = list(df['large_bowel_RLE_encoded'].fillna(""))
    small_bowel_RLE = list(df['small_bowel_RLE_encoded'].fillna(""))
    stomach_RLE = list(df['stomach_RLE_encoded'].fillna(""))
    RLE_masks = list(zip(large_bowel_RLE, small_bowel_RLE, stomach_RLE))

    large_bowel_weight_map = list(df['large_bowel_weighted_map'].fillna(""))
    small_bowel_weight_map = list(df['small_bowel_weighted_map'].fillna(""))
    stomach_weight_map = list(df['small_bowel_weighted_map'].fillna(""))
    weight_maps = list(zip(large_bowel_weight_map, small_bowel_weight_map, stomach_weight_map))

    seg_height = list(df['slice_height'])
    seg_width = list(df['slice_width'])
    seg_shape = list(zip(seg_width, seg_height))

    file_id = list(df['id_x'])
    decode_RLE_encodes_to_single_png(RLE_masks, seg_shape, file_id, 'mask')
    decode_RLE_encodes_to_single_png(weight_maps, seg_shape, file_id, 'map')

    return

def decode_RLE_encodes_to_single_png(encodeds, shapes, file_ids, dtype):
    for encoded, shape, file_id in zip(encodeds, shapes, file_ids):
        masks = []
        for RLE_mask in encoded:
            if RLE_mask != "":
                masks.append(RLE.decode(RLE_mask, shape) * 255)
            else:
                masks.append(np.zeros(shape))
        png_image = np.stack(masks, axis=-1)
        if dtype=='mask':
            filename = os.path.join(Config.MASK_DIR, file_id+'.png')
        elif dtype=='map':
            filename = os.path.join(Config.WEIGHT_MAP_DIR, file_id+'.png')
        
        if not cv2.imwrite(filename, png_image):
            print(filename)
    return


    


def __add_each_weighted_map(df):
    def make_weighted_map(x):
        RLE_encoded, mask_h, mask_w = list(x)
        if RLE_encoded != "":
            a = RLE.decode(RLE_encoded, (mask_h,mask_w))
            b = cv2.erode(a, np.ones((3,3)))
            e = RLE.encode(a-b)
            return e
        return ""
    dff = pd.DataFrame(columns=["id", "large_bowel_weighted_map","small_bowel_weighted_map", "stomach_weighted_map"])
    dff["large_bowel_weighted_map"] = df[['large_bowel_RLE_encoded', 'slice_height', 'slice_width']].apply(make_weighted_map,axis=1)
    dff["small_bowel_weighted_map"] = df[['small_bowel_RLE_encoded', 'slice_height', 'slice_width']].apply(make_weighted_map,axis=1)
    dff["stomach_weighted_map"] = df[['stomach_RLE_encoded', 'slice_height', 'slice_width']].apply(make_weighted_map,axis=1)
    df = pd.merge(df, dff, left_index=True, right_index=True, how='left')
    df.large_bowel_weighted_map = df.large_bowel_weighted_map.fillna("")
    df.small_bowel_weighted_map = df.small_bowel_weighted_map.fillna("")
    df.stomach_weighted_map = df.stomach_weighted_map.fillna("")
    return df












def load_and_preprocess_train_dataframe()->pd.DataFrame:
    train_df = pd.read_csv(Config.TRAIN_CSV)
    all_train_images = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)
    return preprocessing_dataframe(train_df, all_train_images, is_test=False)


def preprocessing_dataframe(df : pd.DataFrame, globbed_full_filepaths : list, is_test=False) -> pd.DataFrame:
    df = __split_case_id_and_add_column(df)
    df = __split_day_number_and_add_column(df)
    df = __split_slice_id_and_add_column(df)
    df = __merge_dataframe_with_filepath(df, globbed_full_filepaths)
    df = __slice_file_height_and_weight(df)
    df = __slice_pixel_spacing(df)

    if not is_test:
        df = __merge_LF_rows_to_single_row_and_multiple_columns(df)
        
    df = __reorder_columns_of_dataframe(df, __get_order_of_columns(is_test))
    return df

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

    df.large_bowel_RLE_encoded = df.large_bowel_RLE_encoded.fillna("")
    df.small_bowel_RLE_encoded = df.small_bowel_RLE_encoded.fillna("")
    df.stomach_RLE_encoded = df.stomach_RLE_encoded.fillna("")
    
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
                     "day_num_str", "day_num", 
                     "slice_id", "predicted"]
    if is_test:
        order_of_column.insert(1, "class")
    return order_of_column

    

