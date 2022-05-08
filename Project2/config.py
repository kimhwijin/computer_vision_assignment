import os
from enums import *
import tensorflow as tf

class Config:

    DATA_DIR:str = os.path.join(os.getcwd(), "row-data")
    TRAIN_DIR:str = os.path.join(DATA_DIR, "train")
    TRAIN_CSV:str = os.path.join(DATA_DIR, "train.csv")
    TEST_DIR:str = os.path.join(DATA_DIR, "test")
    SAVE_DIR:str = os.path.join(DATA_DIR, "save")

    SAMPLE_SUBMIT_CSV:str = os.path.join(DATA_DIR, "sample_submission.csv")
    CLASSES:tuple[str,str,str] = ("Large Bowel", "Small Bowel", "Stomach")
    SF_CLASSES:tuple[str,str,str] = ("lb", "sb", "st")
    SF2LF:dict[str:str] = {_sf:_lf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    LF2SF:dict[str:str] = {_lf:_sf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    NFOLD:int = 8
    IMAGE_SHAPE:tuple[int,int] = (256, 256)
    SEGMENT_SHAPE:tuple[int,int] = (256, 256)
    MASK_STYLE:MASK_STYLE = MASK_STYLE.MULTI_CLASS_MULTI_LABEL
    N_LABELS = 3
    MASK_DIR:str = os.path.join(SAVE_DIR, MASK_STYLE.name, "mask")
    
    class Dataset:
        AUTOTUNE:tf.data.AUTOTUNE = tf.data.AUTOTUNE
        BATCH_SIZE:int = 24
        TRAIN_SHUFFLE_BUFFER_SIZE:int = 30000 // BATCH_SIZE
        VALIDATION_SHUFFLE_BUFFER_SIZE:int = TRAIN_SHUFFLE_BUFFER_SIZE // 5

    class Train:
        inputs = tf.keras.layers.Input((256,256,3))
        kernel_initializer:str=KERNEL_INITIALIZER.HE_NORMAL.value
        activation:str=ACTIVATION.RELU.value

        



