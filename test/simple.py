import cv2
import numpy as np

cap = cv2.VideoCapture('videos/object2.mp4')

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    frame=cv2.rectangle(frame, (100,100),(250,250) , (0,0,0), -1)
    cv2.imshow('frame',frame)

    k=cv2.waitKey(25)
    if k == 27:
        break
    elif k == ord('r'): #리셋
        fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    elif k == ord('l'): #러닝레이트 초기화=>갑작스러운 움직임 만 검출
        lr=-1
    elif k == ord('o'): #본질적으로 모든 움직임 기록
        lr=0

cap.release()
cv2.destroyAllWindows()