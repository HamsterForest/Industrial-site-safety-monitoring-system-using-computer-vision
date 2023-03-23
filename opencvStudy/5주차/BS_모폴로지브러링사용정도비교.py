import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

lr=0

while(1):
    ret, frame = cap.read()

    frame1=frame.copy()

    blur1=cv2.GaussianBlur(frame1,(15,15),5)
    fgmask1=fgbg.apply(blur1,learningRate=lr)
    kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    fgmask1=cv2.morphologyEx(fgmask1,cv2.MORPH_CLOSE,kernel1)
    
    frame2=frame.copy()

    blur2=cv2.GaussianBlur(frame2,(3,3),1)
    fgmask2=fgbg.apply(blur2,learningRate=lr)
    kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
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

    
    cv2.imshow('bluringAndMophology_mask1',fgmask1)
    cv2.imshow('bluringAndMophology1',frame1)
    cv2.imshow('bluringAndMophology_mask2',fgmask2)
    cv2.imshow('bluringAndMophology2',frame2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if k == ord('r'): 
        fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    if k == ord('l'):
        lr=-1
    if k == ord('o'): 
        lr=0

cap.release()
cv2.destroyAllWindows()