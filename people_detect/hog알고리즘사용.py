import cv2
import sys
import random
from imutils.object_detection import non_max_suppression

# 동영상 불러오기
cap = cv2.VideoCapture('videos/vtest.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()
    
# 보행자 검출을 위한 HOG 기술자 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
        
    frame = cv2.resize(frame, (640, 480))

    # 매 프레임마다 보행자 검출
    rects,_ = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.03)
    
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # 검출 결과 화면 표시
    for (x, y, w, h) in pick:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(frame, (x, y, w, h), c, 3)
        
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
