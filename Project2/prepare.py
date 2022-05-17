from typing import List
import pandas as pd
from config import Config
import gc
from utils import RLE
import numpy as np
import os
from preprocess.mask import make_weight_map


def save_weight_map(mask_npy_paths):

    for mask_npy_path in sorted(mask_npy_paths):
        npy_id = mask_npy_path.rsplit('/', 1)[1]
        np_mask = np.load(mask_npy_path)
        np_weight_map = make_weight_map(np_mask)
        np.save(os.path.join(Config.WEIGHT_MAP_DIR, npy_id), np_weight_map)
    
def save_masks(df):
    mask_ids = df.id.to_list()
    
    RLE_large_bowels = df.RLE_large_bowel.to_list()
    RLE_small_bowels = df.RLE_small_bowel.to_list()
    RLE_stomachs = df.RLE_stomach.to_list()
    
    heights = df.height.to_list()
    widths = df.width.to_list()
    n_classes = Config.N_LABELS

    for mask_id, lb, sb, st, height, width in zip(mask_ids, RLE_large_bowels, RLE_small_bowels, RLE_stomachs, heights, widths):
        mask = np.zeros((height, width, n_classes), dtype=np.uint8)
        if lb != "":
            mask[:, :, 0] = RLE.decode(lb, (height, width), np.uint8)
        
        if sb != "":
            mask[:, :, 1] = RLE.decode(sb, (height, width), np.uint8)
        
        if st != "":
            mask[:, :, 2] = RLE.decode(st, (height, width), np.uint8)
        
        
        save_path = os.path.join(Config.MASK_DIR, mask_id + '.npy')
        np.save(save_path, mask)


    
    

def preprocess_csv(image_paths:list):
    df = pd.read_csv(Config.TRAIN_CSV)
    df["case"] = df["id"].apply(lambda x : x.split("_", 2)[0])
    df["day"] = df["id"].apply(lambda x : x.split("_", 2)[1])
    df["slice"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    df["_partial_identifier"] = (Config.TRAIN_DIR + "/" +
                                    df["case"] + "/" + 
                                    df["case"] + "_" + df["day"] + "/"
                                    "scans/" +
                                    df["slice"]
                                )
    _merging_df = pd.DataFrame({
        "_partial_identifier" : [x.rsplit("_", 4)[0] for x in image_paths],
        "full_filepath": image_paths
    })
    df = df.merge(_merging_df, on="_partial_identifier").drop(columns="_partial_identifier")

    df["width"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[2]))

    large_bowel_df = df[df["class"] == "large_bowel"][['id', 'segmentation']].rename(columns={
        "segmentation": "RLE_large_bowel"
    })
    small_bowel_df = df[df["class"] == "small_bowel"][["id", "segmentation"]].rename(columns= {
        "segmentation": "RLE_small_bowel"
    })
    stomach_df = df[df["class"] == "stomach"][["id", "segmentation"]].rename(columns={
        "segmentation": "RLE_stomach"
    })
    df = df.merge(large_bowel_df, on="id", how="left")
    df = df.merge(small_bowel_df, on="id", how="left")
    df = df.merge(stomach_df, on="id", how="left")
    df = df.drop_duplicates(subset=["id", ]).reset_index(drop=True)

    df["large_bowel_flag"] = df["RLE_large_bowel"].apply(lambda x : not pd.isna(x))
    df["small_bowel_flag"] = df["RLE_small_bowel"].apply(lambda x : not pd.isna(x))
    df["stomach_flag"] = df["RLE_stomach"].apply(lambda x : not pd.isna(x))
    df["n_segs"] = df["large_bowel_flag"].astype(int) + df["small_bowel_flag"].astype(int) + df["stomach_flag"].astype(int)
    df = df.drop(columns=["segmentation","class","large_bowel_flag", "small_bowel_flag", "stomach_flag", "case", "day", "slice"])
    df = df.fillna("")
    gc.collect()
    return df

    
