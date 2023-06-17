import cv2
import numpy as np

cap = cv2.VideoCapture('videos/object2.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

lr=0

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    #가우시안 블러링-엣지검출전 노이즈 제거
    blur=cv2.GaussianBlur(frame,(15,15),5)
    #전경마스크 얻기
    fgmask=fgbg.apply(blur,learningRate=lr)
    #모폴로지-타원형 커널
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    #모폴로지-클로징-흰색영역의 검은색 구멍을 메운다
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)


    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    #centroid 무게 중심 좌표
    count_boxes=0
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        
        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:#작은 것은 오류일 가능성이 높다.
            #사각형 그리기
            count_boxes+=1#박스수를 센다
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)#중심부 추적
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))

    cv2.putText(frame,str(count_boxes),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=1)
    cv2.imshow('mask',fgmask)
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