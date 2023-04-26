import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)

lr=0

while(1):
    ret, frame = cap.read()
    
    blur=cv2.GaussianBlur(frame,(15,15),5)
    fgmask=fgbg.apply(blur,learningRate=lr)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    
    fgmask2=fgbg.apply(frame,learningRate=lr)
    frame2=frame.copy()
    frame1=frame.copy()

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
   
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

    
    cv2.imshow('bluringAndMophology_mask',fgmask)
    cv2.imshow('bluringAndMophology',frame1)
    cv2.imshow('with_nothing_mask',fgmask2)
    cv2.imshow('with_nothing',frame2)

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