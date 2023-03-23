import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=250)


while(1):
    ret, frame = cap.read()
    #블러링은 없앰
    #전경마스크 얻기
    fgmask=fgbg.apply(frame)
    #모폴로지-타원형 커널
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
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

    cv2.putText(frame,count_boxes,(0,0),cv2.FONT_HERSHEY_SIMPLEX,3.5,(0,0,255),thickness=2)
    cv2.imshow('mask',fgmask)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('r'): #리셋
        fgbg=cv2.createBackgroundSubtractorMOG2(varThreshold=100)

cap.release()
cv2.destroyAllWindows()