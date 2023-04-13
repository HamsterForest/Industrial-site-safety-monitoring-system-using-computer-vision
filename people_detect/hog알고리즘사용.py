import cv2
import sys
import random

# 동영상 불러오기
cap = cv2.VideoCapture('videos/vtest2.mp4')

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
        
    # 매 프레임마다 보행자 검출
    detected, _ = hog.detectMultiScale(frame) # 사각형 정보를 받아옴
    
    # 검출 결과 화면 표시
    for (x, y, w, h) in detected:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(frame, (x, y, w, h), c, 3)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
