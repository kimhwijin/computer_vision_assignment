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
    t_src_patches = np.transpose(src_patches, (0, 3, 1, 2))
    print(t_src_patches.shape)
    for src_patch in t_src_patches:
        print(src_patch.shape)
        for src_channel in src_patch:
            print(src_channel.shape)
            hist, bin_edges = np.histogram(src_channel, bins=range(0, 255, 5))
            print(hist)
            plt.hist(hist, bin_edges)
            plt.show()