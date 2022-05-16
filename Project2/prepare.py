import pandas as pd
from config import Config
from glob import glob
import os
import gc

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

    df["height"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[1]))
    df["width"] = df["full_filepath"].apply(lambda x : int(x[:-4].rsplit("_", 4)[2]))

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

