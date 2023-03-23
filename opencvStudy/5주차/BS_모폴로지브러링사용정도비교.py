import cv2
import numpy as np
#fgmask2 frame2를 u 를 누를때마다 캡쳐하여 저장하고변수를 정해진 규칙에 따라 변화시킨다.
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
blur_k_size=3
morph_k_size=5
count=1
while(1):
    ret, frame = cap.read()

    frame1=frame.copy()

    blur1=cv2.GaussianBlur(frame1,(15,15),5)
    fgmask1=fgbg.apply(blur1,learningRate=0)#러닝레이트 -1이면, 기준 배경을 천천히 갱신함 0이면 자동으로 갱신 안함
    kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    fgmask1=cv2.morphologyEx(fgmask1,cv2.MORPH_CLOSE,kernel1)
    
    frame2=frame.copy()

    blur2=cv2.GaussianBlur(frame2,(blur_k_size,blur_k_size),(blur_k_size-1)/2)
    fgmask2=fgbg.apply(blur2,learningRate=0)
    kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_k_size,morph_k_size))
    fgmask2=cv2.morphologyEx(fgmask2,cv2.MORPH_CLOSE,kernel2)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask1)
   
    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        
        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame1, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame1, (x, y), (x + width, y + height), (0, 0, 255))

    nlabels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(fgmask2)
   
    for index, centroid in enumerate(centroids2):
        if stats2[index][0] == 0 and stats2[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        
        x, y, width, height, area = stats2[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame2, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame2, (x, y), (x + width, y + height), (0, 0, 255))

    cv2.putText(frame2,str(blur_k_size),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=1)
    cv2.putText(frame2,str(morph_k_size),(90,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=1)

    cv2.imshow('bluringAndMophology_mask1',fgmask1)
    cv2.imshow('bluringAndMophology1',frame1)
    cv2.imshow('bluringAndMophology_mask2',fgmask2)
    cv2.imshow('bluringAndMophology2',frame2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('r'): 
        fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    if k == ord('u'):
        if blur_k_size<15:
            blur_k_size+=2#커널사이즈는 홀수만
        if morph_k_size<51:
            morph_k_size+=2
        cv2.imwrite(f"image_{count}.jpg", frame2)
        count+=1

cap.release()
cv2.destroyAllWindows()