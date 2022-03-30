
## Project 1

Project 목표 : 동일한 객체를 회전시킨 두개의 이미지 1st.jpg, 2nd.jpg 의 대응되는 각 코너를 찾아낸다. <br>

기본 전제
- patch size 는 9x9 이다.
- patch 는 마우스 클릭을 통해 저장한다.
- RGB 3채널 이미지이다.


고안한 방식
- 첫번째 image를 source image, 두번째 이미지를 destination image 로 지정한다.
- 첫번째 image의 patch 는 마우스를 통해서 9x9 패치 4개와 각 패치별 RGB 3채널 즉 4x9x9x3 형태의 패치를 생성한다.
- 두번째 image에서는 마우스 클릭한 부분에 9x9 patch 를 생성하는 대신에 27x27 크기의 넓은 범위의 구간을 지정해 sliding window 형태로 patch 를 생성한다. 즉 27x27 의 구간에서 9x9 패치를 한픽셀씩 움직이면서 총 361개의 patch를 생성한다. 결국 4x361x9x9x3 의 patch 가 만들어진다.
- 첫번째 image의 patch 4개 각각과 두번째 각각 4개의 361x9x9x3 patch 별로 361 개의 loss를 구해 최소 loss 를 찾는다.
- 총 4x4 의 loss 행렬이 만들어지고, 각 값은 첫번째 이미지의 각 patch와 두번째 이미지의 각 대표 patch 사이의 loss 값을 나타낸다. 이때 대표 patch는 361개의 patch 가운데 최소 loss를 가지는 patch 를 의미한다.
- loss 행렬의 argmin by axis=1 을 통해 각 코너에 가장 가깝게 대응되는 코너를 찾는다.

구현 방식
- sliding window 는 window index 를 만들어서 361x4 형태로 x,y와 처음과 끝 인덱스 4개를 저장해서 사용한다.
- 채널은 유지를 하고, 각 채널의 loss 는 평균을 취한다.
- loss 는 각 patch (9x9x3) 들의 pixel histogram (bins-1, 3) 을 구한다. histogram의 bins 는 0 부터 255 까지 block 은 5 pixel 씩으로 지정한다.
- 각 histogram 을 확률로 변화시킨다.
- 첫번째 patch를 통해 구한 histogram propability = p 와 비교할 두번째 histogram propability = q 사이의 KL Divergence 를 통해 loss 를 구한다.
- 첫번째 histogram propability 4x50x3, 두번째 histogram propbability 4x50x3 간의 loss 는 4개의 코너별, 채널별 계산을 수행하므로 총 4x4x3 번의 계산을 수행한다.
- 그렇게 구한 loss 는 4x4 행렬을 띄고, loss[i, j] 는 첫번재 이미지의 i 번째 코너와 두번째 이미지의 j 번재 코너의 histogram probability 의 loss 를 가진다.

실험 방식
- sliding window 를 사용하는 방식과, 그냥 9x9 패치를 사용하는 방식의 차이를 비교한다.
- 두번째 패치를 생성할때, 클릭한 기준으로 27x27 패치와 9x9 패치를 동시에 생성해서 저장한다.
- sliding window 는 위 방색대로 최소 loss 를 구하고, 9x9 패치는 하나이기 때문에 직접 loss를 비교한다.

구현
- main.py
    1. 패치 정보를 가진 src_patches, dst_patches, compare_pathces를 구한다. 각 4x9x9x3, 4x27x27x3, 4x9x9x3 이다.
    2. histogram propability 정보를 가진 src_hists, dst_hists, compare_hists 를 구한다. 각 4x50x3, 4x361x50x3, 4x50x3 이다.
    3. 최소 loss 와 그에 해당하는 histogram 정보를 가진 min_losses, min_hists 를 구한다. 각 4x4, 4x4x50x3 이다.
    4. sliding window 로 구한 loss 와 histogram 을 plot 한다.
    5. 기존 9x9 패치를 사용하여 구한 loss 와 histogram 을 plot 한다.

- patch_generator.py
    - IMAGE_SIZE : 이미지 사이즈
    - SRC_PATCH_SIZE : 첫번째 이미지 patch size -> 9 로 고정이다
    - DST_PATCH_SIZE : 두번째 이미지 검색 구역 size -> 27 로 설정한다. 변경 가능함.
    - get_patches
        1. 두개의 image window 를 생성해 각각 마우스 입력을 받는다.
        2. src_patches, dst_patches, compare_patches 변수를 이용해 정보를 저장 후 리턴한다.
    - check_outline
        1. patch 의 범위가 이미지 크기를 넘어가지 않도록 이미지의 범위를 지정해준다.
- histogram.py
    - BINS : histogram 의 블럭 범위를 지정한다. range(0, 255, 5) 로 지정함.
    - PROP_BINS : 확률화된 histogram 의 블럭 범위를 지정한다. range(0, 1, 1/51) 로 지정함.
    - get_histogram_seperate_channel
        1. window index 를 구한다.
        2. src_patches를 각 채널별로 histogram 을 구하고, 확률화해서 src_hists 를 만든다.
        3. dst_patches를 각 window 별, 채널 별 histogram 을 구하고, 확률화해서 dst_hists 를 만든다.

    - get_histogram_seperate_channel_single_patch
        1. 위의 함수에서 src_patches 가 수행한 부분만 수행한다.
    - build_2d_window_index
        1. src_patch_shape, dst_patch_shape 를 입력받아서 window 개수를 구한다.
        2. window 개수만큼 sliding 하면서 검색 범위의 patch 꼭지점 index 를 저장 후 리턴한다.

- calculation.py
    - kl_divergence
        1. 이산 확률 분포 p, q 를 입력받아서 kl divergence 결과를 구하고 리턴한다.
    - calculation_kl_div_loss
        1. 4x50x3, 4x50x3 와 같이 동일한 shape의 histogram 의 kl-div loss 를 구하고 리턴한다.
    - calculation_kl_div_loss_sliding_window
        1. 4x50x3, 4x361x50x3 shape의 채널 차원기준으로 평균하고, window 차원 기준으로 최소 loss 를 구한다.
        2. 최소 loss 에 해당하는 histogram 과 loss 를 리턴한다.

- visualize.py
    - plot_src_min_hist
        1. 1개의 patch에 해당하는 histogram src_hist 과 그에 해당하는 4개의 histogram 이 저장된 dst_hist, min_loss, src_index 를 입력받는다.
        2. 채널별로 src histogram 과 dst histogram, 총 (1+4)x3 개의 histogram 을 plot 한다.
    - plot_loss
        1. 첫번째 이미지의 patch 순서와 loss 를 입력받아 loss 를 plot 하고, 최소 loss 를 통해서 patch 에 대응되는 patch를 결정한다.


 


