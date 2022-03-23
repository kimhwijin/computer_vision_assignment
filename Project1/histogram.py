import cv2
import numpy as np

clickflag = 0
img1 = cv2.imread('1st.jpg', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, dsize=(300,400))

img2 = cv2.imread('2nd.jpg', cv2.IMREAD_COLOR)
img2 = cv2.resize(img2, dsize=(300,400))


patch1 = []

def draw_rec1(event, x, y, flags, param):
    global patch1
    if event == cv2.EVENT_LBUTTONDOWN:
        patch1.append(cv2.rectangle(img1, (x-51,y-51), (x+51,y+51), (0,255,0), 1)[y-50:y+50, x-50:x+50, :])

def draw_rec2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img2, (x-50,y-50), (x+50,y+50), (0,255,0), 1)

cv2.namedWindow('image1')
cv2.namedWindow('image2')
cv2.setMouseCallback('image1', draw_rec1)
cv2.setMouseCallback('image2', draw_rec2)
while(1):
    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    if cv2.waitKey(20) & 0xFF == 0x1B:
        break




cv2.destroyWindow('image1')
cv2.destroyWindow('image2')


cv2.namedWindow('patch')
while(1):
    cv2.imshow('patch', patch1[0])
    if cv2.waitKey(20) & 0xFF == 0x1B:
        break

cv2.destroyWindow('patch')

import matplotlib.pyplot as plt

for patch in patch1:
    plt.hist(np.reshape(patch, (-1,)))

plt.show()