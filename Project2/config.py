import os

class Config:
    DATA_DIR = os.path.join(os.getcwd(), "dataset")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SAMPLE_SUBMIT_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
    CLASSES = ("Large Bowel", "Small Bowel", "Stomach")
    SF_CLASSES = ("lb", "sb", "st")
    SF2LF = {_sf:_lf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    LF2SF = {_lf:_sf for _sf, _lf in zip(SF_CLASSES, CLASSES)}
    NFOLD = 8
    IMAGE_SHAPE = (256, 256)
    SEGMENT_SHAPE = (256, 256)