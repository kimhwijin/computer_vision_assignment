import numpy as np


def kl_divergence(p, q):
    return np.sum(p * (np.log(p+1e-10) - np.log(q+1e-10)))

def calculate_kl_div_loss(src_hists, dst_hists):
    #src hists : n x bins-1 x c
    #dst hists : n x bins-1 x c

    # n_s x n_d
    losses = np.empty((src_hists.shape[0], dst_hists.shape[0]))
    # n_s x n_d x bins-1 x c
    hists = np.empty((src_hists.shape[0], dst_hists.shape[0], src_hists.shape[1], src_hists.shape[-1]))

    for n_s in range(src_hists.shape[0]):
        for n_d in range(dst_hists.shape[0]):

            loss = 0
            for c in range(src_hists.shape[-1]):
                p = src_hists[n_s, :, c]
                q = dst_hists[n_d, :, c]
                loss += kl_divergence(p, q)
            loss /= 3.
            losses[n_s, n_d] = loss
            hists[n_s, n_d, :, :] = dst_hists[n_d, :, :]

    return losses, hists
    
def calculate_kl_div_loss_sliding_window(src_hists, dst_hists):
    #src hists : n x bins-1 x c
    #dst hists : n x w x bins-1 x c

    # n_s x n_d
    min_losses = np.empty((src_hists.shape[0], dst_hists.shape[0]))
    # n_s x n_d x bins-1 x c
    min_hists = np.empty((src_hists.shape[0], dst_hists.shape[0], src_hists.shape[1], src_hists.shape[-1]))
    #src patch
    for n_s in range(src_hists.shape[0]):
        #dst patch
        for n_d in range(dst_hists.shape[0]):
            
            #dst patch windowes
            #window 을 돌면서 최소 loss 구간을 찾음
            min_loss = 987654321.0
            for w in range(dst_hists.shape[1]):
                loss = 0
                for c in range(3):
                    p = src_hists[n_s,:,c]
                    q = dst_hists[n_d,w,:,c]
                    loss += kl_divergence(p, q)
                loss /= 3
                if loss < min_loss:
                    min_loss = loss
                    min_hist = dst_hists[n_d, w, :, :]
            
            min_hists[n_s, n_d, :, :] = min_hist
            min_losses[n_s, n_d] = min_loss

    #min losses : n_s x n_d
    #min hists : n_s x n_d x bins-1 x c
    return min_losses, min_hists