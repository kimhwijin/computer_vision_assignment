from patch_generator import get_patches
from histogram import *

if __name__ == '__main__':
    src_patch, dst_patch = get_patches()
    get_histogram_seperate_channel(src_patch, dst_patch)
    
