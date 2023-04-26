#함수화로 깔끔한 버전
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
blur_k_size=3
morph_k_size=5
count=1

def process_frame(frame, blur_k_size, morph_k_size):
    blur=cv2.GaussianBlur(frame,(blur_k_size,blur_k_size),(blur_k_size-1)/2)
    fgmask=fgbg.apply(blur,learningRate=0)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_k_size,morph_k_size))
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))

    return fgmask, frame

while(1):
    ret, frame = cap.read()

    fgmask1, frame1 = process_frame(frame.copy(), 15, 51)
    fgmask2, frame2 = process_frame(frame.copy(), blur_k_size, morph_k_size)

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
