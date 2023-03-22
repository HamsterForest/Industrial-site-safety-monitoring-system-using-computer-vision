import cv2 as cv
import numpy as np

img_color=cv.imread("opencvStudy/3.jpg",cv.IMREAD_COLOR)

#이미지의 높이와 너비
height,width=img_color.shape[:2]

#gray scale image를 저장할 numpy 배열을 생성
img_gray=np.zeros((height,width), np.uint8)#8bit unsigned integer

for y in range(0, height):
    for x in range(0, width):
        b=img_color.item(y,x,0)
        g=img_color.item(y,x,1)
        r=img_color.item(y,x,2)

        gray=int(r*0.216+g*0.7152+b*0.0722)
        img_gray.itemset(y,x,gray)

cv.imwrite("opencvStudy/grayscaled_3.jpg",img_gray)
cv.imshow("Gray", img_gray)
cv.waitKey(0)
cv.destroyAllWindows()