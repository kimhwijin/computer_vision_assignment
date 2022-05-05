from enum import Enum, auto
import os
from enums import MASK_STYLE
import tensorflow as tf

class Config:
    DATA_DIR = os.path.join(os.getcwd(), "row-data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SAVE_DIR = os.path.join(DATA_DIR, "save")

    SAMPLE_SUBMIT_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
    CLASSES = ("Large Bowel", "Small Bowel", "Stomach")
    SF_CLASSES = ("lb", "sb", "st")
    SF2LF = {_sf:_lf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    LF2SF = {_lf:_sf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    NFOLD = 8
    IMAGE_SHAPE = (256, 256)
    SEGMENT_SHAPE = (256, 256)
    MASK_STYLE = MASK_STYLE.MULTI_CLASS_ONE_LABEL
    MASK_DIR = os.path.join(SAVE_DIR, MASK_STYLE.name, "mask")
    
    class Dataset:
        AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE = 24
        SHUFFLE_BUFFER_SIZE = max(BATCH_SIZE*25, 500)

    

