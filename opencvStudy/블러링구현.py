import cv2
import numpy as np

image = cv2.imread('opencvStudy/3.jpg')
image2 = image.copy()
blur_bycv = cv2.blur(image2,(5,5))

height, width, channels = image.shape

#마스크의 사이즈
kernel_size = 5

kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
#np.ones는 1로 채워진 매트릭스를 만드는 함수다. dtype에서 커널의 값이 32bit float 자료형이 되도록 한다.
#마지막으로 커널사이즈의 제곱으로 나누는데 그러면, 각 배열의 원소로, 커널 전체 넓이의 역수가 들어간다.
#이는 평균값 필터의 핵심으로 커널의 가운데에 있는 픽셀을 중심으로 커널 안에 들어가는 모든 픽셀의 평균을 구하게 된다.
#예를들어
# 1 4 5
# 4 3 2
# 1 2 5 라는 배열에 1/9를 각각 곱하고 이들을 모두 더하면 3x3 이미지의 평균 픽셀값이 도출된다.
# 그 값을 가운데 예시에서는 (1,1)애 대입한다.

blurred_image = np.copy(image)

# 가장자리부분은 마스크가 안 들어감으로. kernel_size//2 는 커널 중심점에서 모서리까지의 거리
for y in range(kernel_size//2, height-kernel_size//2):
    for x in range(kernel_size//2, width-kernel_size//2):
        
        #블러링 마스크를 적용할 지역을 원본이미지에서 따온다. region of interest 
        roi = image[y-kernel_size//2:y+kernel_size//2+1, x-kernel_size//2:x+kernel_size//2+1]
    
        #만들어 놓은 커널 마스크를 적용한다.
        #[0,0,0] 생성 r,g,b
        blurred_value = np.zeros((3,), dtype=np.float32)
        #채널은 3개 r,g,b
        for c in range(channels):
            #roi는 3차원 배열이다. 한 픽셀 안의 어떤 색요소만을 추출함으로 roi[:,:,c]
            #한 특정 색마다 커널 마스크를 곱해준다.
            #axis=(0,1) 합연산을 1,2차원에서 진행한다.
            #마스크와 roi 의 c값을 곱하고 이 값을 모두 더함.
            blurred_value[c] = np.sum(kernel * roi[:,:,c], axis=(0, 1))
        
        #대입
        blurred_image[y, x] = blurred_value

cv2.imshow('original Image', image)
cv2.imshow('blur_by_cv2',blur_bycv)
cv2.imshow('blurred Image', blurred_image)
cv2.waitKey(0)
