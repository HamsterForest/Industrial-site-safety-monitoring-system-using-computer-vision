import cv2
import numpy as np
import time

cap = cv2.VideoCapture('videos/object3.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

lr = 0

objects = {}
objects_duration = {}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))

    blur = cv2.GaussianBlur(frame, (15, 15), 5)
    fgmask = fgbg.apply(blur, learningRate=lr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

    count_boxes = 0
    current_time = time.time()

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        
        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 100:
            #오브젝트 안에 인덱스 존재
            if index in objects:
                #abs 절대값 함수, 오자범위 안에 있는지 확인 사이즈 40% 중심점 흔들림 20% 기준
                if abs(width - objects[index][0]) / objects[index][0] <= 0.4 and \
                   abs(height - objects[index][1]) / objects[index][1] <= 0.4 and \
                   abs(centerX - objects[index][2]) / objects[index][2] <= 0.2 and \
                   abs(centerY - objects[index][3]) / objects[index][3] <= 0.2:
                    # 3초이상 적치시 파란색으로 경계 바꾸어 그림 아니면 빨간색
                    if current_time - objects_duration[index] >= 3:
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                else:
                    # 오브젝트 사이즈와 중심점 좌표, 시간 업데이트(오차범위 안에 없으니 시간 초기화)
                    objects[index] = [width, height, centerX, centerY]
                    objects_duration[index] = current_time
            else:
                # 새로운 오브젝트 추가
                objects[index] = [width, height, centerX, centerY]
                objects_duration[index] = current_time

            count_boxes += 1
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)

    cv2.putText(frame, str(count_boxes), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
    cv2.imshow('mask', fgmask)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(25)
    if k == 27:
        break
    elif k == ord('r'):
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)
    elif k == ord('l'):
        lr = -1
    elif k == ord('o'):
        lr = 0

cap.release()
cv2.destroyAllWindows()
