import os
from config import Config
import pandas as pd
from glob import glob
import numpy as np

train_df = pd.read_csv(Config.TRAIN_CSV)
all_train_images = glob(os.path.join(Config.TRAIN_DIR, "**", "*.png"), recursive=True)

ss_df = pd.read_csv(Config.SAMPLE_SUBMIT_CSV)