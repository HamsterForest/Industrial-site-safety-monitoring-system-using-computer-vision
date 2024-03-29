import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

lr=0

while(1):
    ret, frame = cap.read()
    #가우시안 블러링-엣지검출전 노이즈 제거
    blur=cv2.GaussianBlur(frame,(15,15),5)
    #전경마스크 얻기
    fgmask1=fgbg.apply(blur,learningRate=lr)
    fgmask2=fgbg.apply(frame,learningRate=lr)


    cv2.imshow('mask_blur',fgmask1)
    cv2.imshow('no_blur',fgmask2)
    

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('r'): #리셋
        fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    if k == ord('l'): #러닝레이트 초기화=>갑작스러운 움직임 만 검출
        lr=-1
    if k == ord('o'): #본질적으로 모든 움직임 기록
        lr=0

cap.release()
cv2.destroyAllWindows()