from skimage.segmentation import find_boundaries
import numpy as np

def make_weight_map(masks:np.ndarray):
    """
    masks : height, width, n classes
    """
    h, w = masks.shape[:2]
    n_classes = masks.shape[-1]
    masks = (masks > 0. ).astype(np.uint8)
    distance_map = np.zeros((h*w, n_classes))
    X1, Y1 = np.meshgrid(np.arange(h), np.arange(w))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T





