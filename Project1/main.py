from calculation import *
from patch_generator import get_patches
from histogram import *
from visualize import *

if __name__ == '__main__':
    src_patches, dst_patches = get_patches()
    src_hists, dst_hists = get_histogram_seperate_channel(src_patches, dst_patches)


    #src_histes : n x bins-1 x c
    #dst_histest : n x w x bins-1 x c
    min_losses, min_hists = calculate_kl_div_loss(src_hists, dst_hists)

    for s in range(4):
        plot_src_min_hist(src_hists[s], min_hists[s], min_losses[s], s+1)
        
    print(np.argmin(min_losses, axis=1)+1)

    



    
