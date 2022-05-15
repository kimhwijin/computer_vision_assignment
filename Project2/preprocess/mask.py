from skimage.segmentation import find_boundaries
import numpy as np

def make_weight_map(masks:np.ndarray, w0=10, sigma=5):
    """
    masks : height, width, n classes
    """
    #masks type -> binary uint8
    masks = ( masks > 0 ).astype(np.uint8)
    # 비어있는 마스크를 제거함
    masks = masks[:,:,np.argwhere(masks.sum(axis=(0, 1)) > 0).reshape(-1)]

    # masks = height, width, channel
    masks = masks.transpose([2,0,1])
    # masks = channel, height, width
    
    h, w = masks.shape[1:]
    n_classes = masks.shape[0]
    if n_classes == 0:
        return np.zeros((h, w))
    distance_map = np.zeros((h*w, n_classes))

    X1, Y1 = np.meshgrid(np.arange(h), np.arange(w))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T

    for i, mask in enumerate(masks):
        #오브젝트의 바운더리를 구함
        bounds = find_boundaries(mask, mode='inner')
        #바운더리 픽셀만 남겨둠
        X2, Y2 = np.nonzero(bounds)
        #바운더리 픽셀과 나머지 모든 픽셀간의 각 축에 대한 거리 제곱을 구함
        x_sum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        y_sum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        #각 픽셀 위치에서 가장 가까운 바운더리 픽셀을 정함.
        
        distance_map[:, i] = np.sqrt(x_sum + y_sum).min(axis=0)
        

    idx = np.arange(distance_map.shape[0])
    if distance_map.shape[1] == 1:
        d1 = distance_map.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1)**2) / (2 * (sigma**2)))
    else:
        if distance_map.shape[1] == 2:
            d1_idx, d2_idx = np.argpartition(distance_map, 1, axis=1)[:, :2].T
        else:
            # 각 픽셀에 대해서 오브젝트간의 거리중 가장 가까운 것과 두번째로 가까운것 을 취함.
            d1_idx , d2_idx = np.argpartition(distance_map, 2, axis=1)[:, :2].T
        # 각 픽셀의 오브젝트 거리중 가장 가까운 것과 두번째로 가까운 거리값들을 추출하는 과정
        d1 = distance_map[idx, d1_idx]
        d2 = distance_map[idx, d2_idx]
        #바운더리 loss 맵 2번째항 계산
        border_loss_map = w0 * np.exp((-1 * (d1 + d2)**2) / (2 * (sigma**2)))
    
    # 두번째 바운더리 거리 맵
    xBLoss = np.zeros((h, w))
    xBLoss[X1, Y1] = border_loss_map

    # 첫번째 frequencies 밸런스 맵
    loss = np.zeros((h, w))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0

    # 위의 두 항을 더함
    ZZ = xBLoss + loss
    return ZZ