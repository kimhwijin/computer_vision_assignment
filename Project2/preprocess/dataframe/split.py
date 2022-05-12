from typing import Tuple
from sklearn.model_selection import GroupKFold
from config import Config
import pandas as pd 

def create_kfold_train_validation_dataframe(df:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
    df = __encode_segment_flag_columns_to_single_string_column(df)
    train_df, valid_df = __split_kfold_train_validation_and_sampling_dataframe(df)
    return train_df, valid_df

def __encode_segment_flag_columns_to_single_string_column(df:pd.DataFrame)->pd.DataFrame:
    df["which_segments"] = df["large_bowel_flag"].astype(int).astype(str) \
                            + df["small_bowel_flag"].astype(int).astype(str) \
                            + df["stomach_flag"].astype(int).astype(str)
    return df

def __split_kfold_train_validation_and_sampling_dataframe(df:pd.DataFrame)->Tuple[pd.DataFrame, pd.DataFrame]:
    group_kfold = GroupKFold(n_splits=Config.NFOLD)

    for train_idxs, valid_idxs in group_kfold.split(df["id"], df["which_segments"], df["case_id"]):
        
        fold_train_df = df.iloc[train_idxs]
        N_TRAIN = len(fold_train_df)
        fold_train_df = fold_train_df.sample(N_TRAIN).reset_index(drop=True)

        fold_valid_df = df.iloc[valid_idxs]
        N_VALID = len(fold_valid_df)
        fold_valid_df = fold_valid_df.sample(N_VALID).reset_index(drop=True)
        break
    
    return fold_train_df, fold_valid_df

