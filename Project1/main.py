from calculation import kl_divergence
from patch_generator import get_patches
from histogram import *

if __name__ == '__main__':
    src_patches, dst_patches = get_patches()
    src_histes, dst_histes = get_histogram_seperate_channel(src_patches, dst_patches)


    #src_histes : n x bins-1 x c
    #dst_histest : n x w x bins-1 x c

    min_losses = np.empty((src_histes.shape[0], dst_histes.shape[0]))

    #src patch
    for n_s in range(src_histes.shape[0]):
        #dst patch
        for n_d in range(dst_histes.shape[0]):
            
            #dst patch windowes
            #window 을 돌면서 최소 loss 구간을 찾음
            min_loss = 987654321.0
            for w in range(dst_histes.shape[1]):
                loss = 0
                for c in range(3):
                    p = src_histes[n_s,:,c]
                    q = dst_histes[n_d,w,:,c]
                    loss += kl_divergence(p, q)
                loss /= 3
                if loss < min_loss:
                    min_loss = loss
            
            min_losses[n_s, n_d] = min_loss

    print(min_losses)
    print(np.argmin(min_losses, axis=1)+1)
    # src_hist = src_histes[0]
    # plt.plot(range(0, 255, 5)[:-1], src_hist[:, 0],'r')
    # plt.plot(range(0, 255, 5)[:-1], src_hist[:, 1], 'g')
    # plt.plot(range(0, 255, 5)[:-1], src_hist[:, 2], 'b')


    # dst_hist = dst_histes[0, 0]
    # plt.plot(range(0, 255, 5)[:-1], dst_hist[:, 0],'yellow')
    # plt.plot(range(0, 255, 5)[:-1], dst_hist[:, 1], 'pink')
    # plt.plot(range(0, 255, 5)[:-1], dst_hist[:, 2], 'purple')
    # plt.show()
    



    
