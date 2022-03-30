import numpy as np
import matplotlib.pyplot as plt

BINS = range(0, 255, 5)
PROP_BINS = np.arange(0, 1, 1/51)

def get_histogram(src_patches, dst_patches):
    #src patch : n x 9 x 9 x 3 ( expect n = 4 )
    #dst patch : n x 27 x 27 x 3 ( expect n = 4 )
    for src_patch in src_patches:
        print(src_patch.shape)
        hist, bin_edges = np.histogram(src_patch, bins=BINS)
        plt.hist(hist, bin_edges)
        plt.show()

def get_histogram_seperate_channel_single_patch(patches):
    n_channel = patches.shape[-1]
    hists = np.empty((patches.shape[0], len(BINS)-1, n_channel), np.float32)

    for n, patch in enumerate(patches):
        for c in range(n_channel):
            hist, _ = np.histogram(patch[:, :,c], bins=BINS)
            hists[n, :, c] = hist / np.sum(hist)

    return hists


def get_histogram_seperate_channel(src_patches, dst_patches):
    # n x 9 x 9 x 3
    # n x 27 x 27 x 3

    window_index = build_2d_window_index(src_patches.shape, dst_patches.shape)

    # n, bins-1, c
    src_hist = np.empty((src_patches.shape[0], len(BINS) - 1, src_patches.shape[-1]), np.float32)
    # w, n, bins-1, c
    dst_hist = np.empty((dst_patches.shape[0], len(window_index), len(BINS) - 1, dst_patches.shape[-1]), np.float32)

    t_src_patches = np.transpose(src_patches, (0, 3, 1, 2))
    
    for n, src_patch in enumerate(t_src_patches):
        for c, src_channel in enumerate(src_patch):
            hist, _ = np.histogram(src_channel, bins=BINS)
            src_hist[n,:,c] = hist / np.sum(hist)

    t_dst_patches = np.transpose(dst_patches, (0, 3, 1, 2))

    for n, dst_patch in enumerate(t_dst_patches):
        for c, dst_channel in enumerate(dst_patch):
            for w, (h_s, h_e, w_s, w_e) in enumerate(window_index):
                hist, _ = np.histogram(dst_channel[h_s:h_e,w_s:w_e], bins=BINS)
                dst_hist[n, w, :, c] = hist / np.sum(hist)

    print(t_src_patches.shape, t_dst_patches.shape)
    print(src_hist.shape, dst_hist.shape)

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
        




    

