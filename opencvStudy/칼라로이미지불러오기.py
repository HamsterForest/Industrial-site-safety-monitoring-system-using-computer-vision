import cv2

img_color=cv2.imread('opencvStudy/3.jpg',cv2.IMREAD_COLOR)
cv2.namedWindow("cup3")
cv2.imshow('cup3',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
