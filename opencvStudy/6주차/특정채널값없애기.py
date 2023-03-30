import cv2

image = cv2.imread('opencvStudy/3.jpg')

height, width= image.shape[0:2]
image1=image.copy()
image2=image.copy()
image3=image.copy()

for y in range(height):
    for x in range(width):
        blue, green, red = image[y, x]
        #블루 채널 제거
        blue = 0
        #업데이트
        image1[y, x] = [blue, green, red]

# Display the new image
cv2.imshow('New Image', image1)
cv2.waitKey(0)

for y in range(height):
    for x in range(width):
        blue, green, red = image[y, x]
        #그린 채널 제거
        green = 0
        #업데이트
        image2[y, x] = [blue, green, red]

# Display the new image
cv2.imshow('New Image', image2)
cv2.waitKey(0)

for y in range(height):
    for x in range(width):
        blue, green, red = image[y, x]
        #레드 채널 제거
        red = 0
        #업데이트
        image3[y, x] = [blue, green, red]

# Display the new image
cv2.imshow('New Image', image3)
cv2.waitKey(0)

