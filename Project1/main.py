from calculation import *
from patch_generator import get_patches
from histogram import *
from visualize import *

if __name__ == '__main__':
    src_patches, dst_patches, compare_patches = get_patches()

    #src_histes : n x bins-1 x c
    #dst_histest : n x w x bins-1 x c
    src_hists, dst_hists = get_histogram_seperate_channel(src_patches, dst_patches)
    #compare_histes : n x bins-1 x c
    compare_hists = get_histogram_seperate_channel_single_patch(compare_patches)

    
    # sliding windows
    min_losses, min_hists = calculate_kl_div_loss_sliding_window(src_hists, dst_hists)
    for s in range(4):
        plot_src_min_hist(src_hists[s], min_hists[s], min_losses[s], s+1)

    for s in range(4):
        plot_loss(min_losses[s], s+1)
    
    # compare
    losses, hists = calculate_kl_div_loss(src_hists, compare_hists)
    for s in range(4):
        plot_src_min_hist(src_hists[s], hists[s], losses[s], s+1)
    for s in range(4):
        plot_loss(losses[s], s+1)

    print(np.argmin(min_losses, axis=1)+1)

    



    
