import enum
from operator import mod
import numpy as np
import matplotlib.pyplot as plt

def get_histogram(src_patches, dst_patches):
    #src patch : n x 9 x 9 x 3 ( expect n = 4 )
    #dst patch : n x 27 x 27 x 3 ( expect n = 4 )
    for src_patch in src_patches:
        print(src_patch.shape)
        hist, bin_edges = np.histogram(src_patch, bins=range(0, 255, 5))
        plt.hist(hist, bin_edges)
        plt.show()

def get_histogram_seperate_channel(src_patches, dst_patches):
    # n x 9 x 9 x 3
    # n x 27 x 27 x 3

    window_index = build_2d_window_index(src_patches.shape, dst_patches.shape)

    bins = range(0, 255, 5)
    # n, bins-1, c
    src_hist = np.empty((src_patches.shape[0], len(bins) - 1, src_patches.shape[-1]), np.float32)
    # w, n, bins-1, c
    dst_hist = np.empty((len(window_index), dst_patches.shape[0], len(bins) - 1, dst_patches.shape[-1]), np.float32)

    t_src_patches = np.transpose(src_patches, (0, 3, 1, 2))

    for n, src_patch in enumerate(t_src_patches):
        for c, src_channel in enumerate(src_patch):
            src_hist[n,:,c], _ = np.histogram(src_channel, bins=bins)

    t_dst_patches = np.transpose(dst_patches, (0, 3, 1, 2))
    
    for n, dst_patch in enumerate(t_dst_patches):
        for c, dst_channel in enumerate(dst_patch):
            for w, (h_s, h_e, w_s, w_e) in enumerate(window_index):
                dst_hist[w, n, :, c], _ = np.histogram(dst_channel[h_s:h_e,w_s:w_e], bins=bins)

    return src_hist, dst_hist

def build_2d_window_index(src_patch_shape, dst_patch_shape):
    n_windows = (dst_patch_shape[1] - src_patch_shape[1] + 1) * (dst_patch_shape[2] - src_patch_shape[2] + 1)
    window_index = []
    #0, 27, 0, 27
    h_s, h_e, w_s, w_e = 0, src_patch_shape[1], 0, src_patch_shape[2]

    for _ in range(n_windows):
        window_index.append((h_s, h_e, w_s, w_e))

        if w_e == dst_patch_shape[2]:
            w_s = 0
            w_e = src_patch_shape[2]
            h_s += 1
            h_e += 1
        else:
            w_s += 1
            w_e += 1
    
    return window_index
        


    

