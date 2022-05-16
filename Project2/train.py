from click import FileError
from config import Config
import numpy as np
from glob import glob
import os

def get_path_datasets():

    image_paths = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)
    image_paths = sorted(image_paths)

    mask_paths = glob(os.path.join(Config.MASK_DIR, "*.png"))
    mask_paths = sorted(mask_paths)

    weight_map_paths = glob(os.path.join(Config.WEIGHT_MAP_DIR, "*.png"))
    weight_map_paths = sorted(weight_map_paths)

    if not (len(image_paths) == len(mask_paths) == len(weight_map_paths)):
        raise Exception("image, mask, weight map datas not equal")
    
    dataset = np.stack([image_paths,mask_paths,weight_map_paths], axis=1)
    np.random.shuffle(dataset)

    return dataset

    

