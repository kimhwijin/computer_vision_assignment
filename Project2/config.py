import os
from typing import Tuple
from enums import *
import tensorflow as tf

class Config:
    
    # DATA_DIR:str = os.path.join(os.getcwd(), "row-data")
    DATA_DIR:str = os.path.join('/content', 'computer_vision_assignment', 'Project2', "row-data")
    TRAIN_DIR:str = os.path.join(DATA_DIR, "train")
    TRAIN_CSV:str = os.path.join(DATA_DIR, "train.csv")
    TEST_DIR:str = os.path.join(DATA_DIR, "test")
    SEGMENT_DIR:str = os.path.join(DATA_DIR, "segmentation")
    MASK_DIR:str = os.path.join(SEGMENT_DIR, "mask")
    WEIGHT_MAP_DIR:str = os.path.join(SEGMENT_DIR, "weight_map")

    TRAINABLE_CSV:str = os.path.join(DATA_DIR, "trainable.csv")
    SAMPLE_SUBMIT_CSV:str = os.path.join(DATA_DIR, "sample_submission.csv")
    CLASSES:Tuple[str,str,str] = ("Large Bowel", "Small Bowel", "Stomach")
    SF_CLASSES:Tuple[str,str,str] = ("lb", "sb", "st")
    SF2LF = {_sf:_lf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    LF2SF = {_lf:_sf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    NFOLD:int = 8
    IMAGE_SHAPE:Tuple[int,int] = (256, 256)
    SEGMENT_SHAPE:Tuple[int,int] = (256, 256)
    MASK_STYLE:MASK_STYLE = MASK_STYLE.MULTI_CLASS_MULTI_LABEL
    N_LABELS = 3
    ALREADY_SAVED_MASK:bool = True
    ALREADY_SAVED_WEIGHT_MAP:bool = True
    

    class Dataset:
        AUTOTUNE:tf.data.AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE:int = 16
        TRAIN_SHUFFLE_BUFFER_SIZE:int = 30000 // BATCH_SIZE
        VALIDATION_SHUFFLE_BUFFER_SIZE:int = TRAIN_SHUFFLE_BUFFER_SIZE // 5
        SEED = 1004

    class Train:
        inputs = tf.keras.layers.Input((256,256,1))
        kernel_initializer:str=KERNEL_INITIALIZER.HE_NORMAL.value
        activation:str=ACTIVATION.RELU.value

        



