import cv2 as cv
img_color=cv.imread("opencvStudy/3.jpg",cv.IMREAD_COLOR)
print(img_color.shape)

#이미지의 높이와 너비
height,width=img_color.shape[:2]
