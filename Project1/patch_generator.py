import glob
import cv2
import numpy as np
from functools import partial

IMAGE_SIZE = (300, 400)
SRC_PATCH_SIZE = 9
DST_PATCH_SIZE = 27

def get_patches(src_patch_size=SRC_PATCH_SIZE, dst_patch_size=DST_PATCH_SIZE):
    image_paths = glob.glob("*.jpg")
    src_image, dst_image = load_and_resize_image(image_paths)
    src_patches, dst_patches = [], []
    draw_src_img = partial(draw_rec, img=src_image, patch=src_patches, patch_size=SRC_PATCH_SIZE)
    draw_dst_img = partial(draw_rec, img=dst_image, patch=dst_patches, patch_size=DST_PATCH_SIZE)
    cv2.namedWindow('src_image')
    cv2.namedWindow('dst_image')
    cv2.setMouseCallback('src_image', draw_src_img)
    cv2.setMouseCallback('dst_image', draw_dst_img)
    while(1):
        cv2.imshow('src_image', src_image)
        cv2.imshow('dst_image', dst_image)
        if cv2.waitKey(20) & 0xFF == 0x1B:
            break
    cv2.destroyAllWindows()
    return np.array(src_patches), np.array(dst_patches)

def load_and_resize_image(image_paths):
    imgs = []
    for image_path in image_paths:
        imgs.append(cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), IMAGE_SIZE))
    return imgs

def check_outline(x, y, patch_size):
    max_x, max_y = IMAGE_SIZE
    if x - patch_size // 2 < 0:
        x_1 = 0
        x_2 = patch_size
    elif (x + patch_size // 2) >= max_x:
        x_1 = max_x - patch_size
        x_2 = max_x
    else:
        x_1 = x - patch_size // 2
        x_2 = x + patch_size // 2 + 1
    
    if y - patch_size // 2 < 0:
        y_1 = 0
        y_2 = patch_size
    elif (y + patch_size // 2) >= max_y:
        y_1 = max_y - patch_size
        y_2 = max_y
    else:
        y_1 = y - patch_size // 2
        y_2 = y + patch_size // 2 + 1
    return (x_1, x_2) , (y_1, y_2)


def draw_rec(event, x, y, flags, param, img, patch, patch_size):
    if event == cv2.EVENT_LBUTTONDOWN:
        (x_1, x_2), (y_1, y_2) = check_outline(x, y, patch_size)
        patch.append(cv2.rectangle(img, (x_1-1,y_1-1), (x_2,y_2), (0,255,0), 1)[y_1:y_2, x_1:x_2, :])

        cv2.putText(img, "1", (x_1-1, y_1-1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0,0,155), 2, cv2.LINE_AA)
        print("added patch")