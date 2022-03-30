import matplotlib.pyplot as plt
from histogram import PROP_BINS
import numpy as np


def plot_src_min_hist(src_hist, min_hist, min_loss, src_index):
    """
    src_hist : bins-1 x c
    min_hist : n_d x bins-1 x c
    min_loss : n_d
    """

    min_loss_arg = np.argmin(min_loss, axis=0)+1

    plt.figure(figsize=(10, 6))
    plt.title("src hist and min hists")
    plt.ylabel("channels 1 2 3")
    plt.xlabel("src hist - min hists 1 ~ 4")
    for c in range(3):
        plt.subplot(3, 5, c*5+1)
        plt.title("src hist {}".format(src_index))
        plt.hist(src_hist[:, c], PROP_BINS)
        for d in range(4):
            plt.subplot(3, 5, c*5+d+2)
            if min_loss_arg == d + 1:
                plt.title("min loss hist {:.3f}".format(min_loss[min_loss_arg-1]))
            plt.hist(min_hist[d, :, c], PROP_BINS)
    
    plt.show()
    