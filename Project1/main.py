from patch_generator import get_patches
from histogram import *

if __name__ == '__main__':
    src_patches, dst_patches = get_patches()
    src_hist, dst_hist = get_histogram_seperate_channel(src_patches, dst_patches)
    print(src_hist.shape, dst_hist.shape)

    
