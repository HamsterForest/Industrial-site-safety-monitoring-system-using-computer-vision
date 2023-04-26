#모폴로지 침식 구현
#모폴로지 침심의 원리는 다음과 같다.
#먼저 structure element라고 하는 구조요소를 선언한다. 여기서는 커널로 표현된다.
#이 구조요소의 모양은 십자가, 타원, 정방형이 있고 여기서는 정방형을 사용한다.
#이미지를 픽셀단위로 구조요소를 대입한다. 그러면 목표 픽셀이 구조요소의 정중앙에 있게 된다.
#이 상태에서 구조요소 안의 픽셀들을 모두 검사하여 가장 작은 값을 찾는다. 그리고 목표 픽셀에 그 값을 대입한다.
#이 과정을 모든 픽셀에 적용하면 침식이 일어나게 된다.
#모난 부분이 깍여나가고 이미지 안의 형태들이 침식해 나가게 되는 것으로 보여진다.

#모폴로지의 팽창은 침식과 반대로 최대값을 찾게 된다.
#모폴로지의 클로징과 오프닝은 침식과 팽창을 다른 순서로 순서대로 진행시켜서 얻는다.
#오프닝 : 침식->팽창
#침식을 먼저 하기 때문에 한두픽셀 짜리 영역이 제거된 후 팽창연산이 진행되고, 영상에 존재하는 작은 크기의 객체가
#사라진다.
#클로징 : 팽창->침식
#팽창연산을 먼저 수행하여 내부의 작은 구멍이 메워진후 침식이 진행되고, 결과적으로 작은 구멍이 제거된다.


import numpy as np
import cv2

def erosion(img, kernel_size):
    #img와 같은 크기를 가지는 0 배열 생성
    output = np.zeros_like(img)
    
    #이미지 크기
    rows, cols = img.shape
    
    for i in range(rows):
        for j in range(cols):
            #최소값을 가능한 최대값으로 선언
            min_val = 255
            
            for m in range(kernel_size):
                for n in range(kernel_size):
                    row = i - kernel_size // 2 + m
                    col = j - kernel_size // 2 + n
                    
                    if row >= 0 and col >= 0 and row < rows and col < cols:
                    
                        if img[row, col] < min_val:
                            #최소값 갱신
                            min_val = img[row, col]
            
            #최소값 대입
            output[i, j] = min_val
    
    return output

#그레이스케일로 불러온다.
image = cv2.imread('opencvStudy/3.jpg', cv2.IMREAD_GRAYSCALE)
image2 = image.copy()
eroded_by_cv2 = cv2.morphologyEx(image2, cv2.MORPH_ERODE,kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
eroded_img = erosion(image, 7)

cv2.imshow('Original Image', image)
cv2.imshow('Eroded by cv2', eroded_by_cv2)
cv2.imshow('Eroded Image', eroded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
