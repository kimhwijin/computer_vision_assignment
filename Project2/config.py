import os
from typing import Tuple
import enums
import tensorflow as tf
import sys

class Config:
    
    DATA_DIR:str = os.path.join(os.getcwd(), "row-data")
    IS_COLAB = "google.colab" in sys.modules
    if IS_COLAB:
        DATA_DIR:str = os.path.join('/content', 'computer_vision_assignment', 'Project2', "row-data")
        
    TRAIN_DIR:str = os.path.join(DATA_DIR, "train")
    TRAIN_CSV:str = os.path.join(DATA_DIR, "train.csv")
    
    TEST_DIR:str = os.path.join(DATA_DIR, "test")
    if not os.path.exists(TEST_DIR): os.mkdir(TEST_DIR)

    SEGMENT_DIR:str = os.path.join(DATA_DIR, "segmentation")
    if not os.path.exists(SEGMENT_DIR): os.mkdir(SEGMENT_DIR)

    MASK_DIR:str = os.path.join(SEGMENT_DIR, "mask")
    if not os.path.exists(MASK_DIR): os.mkdir(MASK_DIR)

    WEIGHT_MAP_DIR:str = os.path.join(SEGMENT_DIR, "weight_map")
    if not os.path.exists(WEIGHT_MAP_DIR): os.mkdir(WEIGHT_MAP_DIR)

    SAMPLE_SUBMIT_CSV:str = os.path.join(DATA_DIR, "sample_submission.csv")
    CLASSES:Tuple[str,str,str] = ("Large Bowel", "Small Bowel", "Stomach")
    SF_CLASSES:Tuple[str,str,str] = ("lb", "sb", "st")
    SF2LF = {_sf:_lf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    LF2SF = {_lf:_sf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    NFOLD:int = 8
    IMAGE_SHAPE:Tuple[int,int] = (256, 256)
    SEGMENT_SHAPE:Tuple[int,int] = (256, 256)
    N_LABELS = 3
    MASK_STYLE:enums.MASK_STYLE = enums.MASK_STYLE.MULTI_CLASS_MULTI_LABEL
    ALREADY_SAVED_MASK:bool = True
    ALREADY_SAVED_WEIGHT_MAP:bool = True
    

    SEED = 1004
    AUTOTUNE:tf.data.AUTOTUNE = tf.data.AUTOTUNE
    BATCH_SIZE:int = 32
    TRAIN_SHUFFLE_BUFFER_SIZE:int = 30000 // BATCH_SIZE
    VALIDATION_SHUFFLE_BUFFER_SIZE:int = TRAIN_SHUFFLE_BUFFER_SIZE // 5
    
    MODEL_IMAGE_INPUT_SHAPE = (*IMAGE_SHAPE, 1)
    MODEL_WEIGHT_MAP_INPUT_SHAPE = (*IMAGE_SHAPE, 1)
    
    KERNEL_INITIALIZER:str= enums.KERNEL_INITIALIZER.HE_NORMAL.value
    activation:str=enums.ACTIVATION.RELU.value

        



