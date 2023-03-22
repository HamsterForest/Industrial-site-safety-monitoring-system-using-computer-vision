import cv2 as cv
import numpy as np
import os

cap=cv.VideoCapture(0)

#history는 과거 몇 프레임을 배경으로 사용할지에 대한 것
#두번째는 마할라노비스 거리의 제곱에 대한 임계값 
#마할라노비스 거리: 주어진 데이터들의 분포를 통해 맥락을 조사하고, 이를 정규화 한 뒤에 유클리드 거리를 계산
#세 번째는 그림자 검출 여부
foregroundBackground=cv.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

while(1):
    ret,img_frame=cap.read() #ret는 제대로 읽어오면 true를 반환함
    if ret==False:
        break
    
    #가우시안 블러링-엣지검출전 노이즈 제거
    blur=cv.GaussianBlur(img_frame,(5,5),0)

    #전경마스크 얻기
    img_mask=foregroundBackground.apply(blur,learningRate=0)
    #모폴로지-타원형 커널
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
    #모폴로지-클로징-흰색영역의 검은색 구멍을 메운다
    img_mask=cv.morphologyEx(img_mask,cv.MORPH_CLOSE,kernel)

    cv.imshow('mask',img_mask)
    cv.imshow('color',img_frame)

    key=cv.waitKey(30)
    if key == 27:
        break
    if key == ord('r'): #리셋
        foregroundBackground=cv.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

cap.release()
cv.destroyAllWindows()