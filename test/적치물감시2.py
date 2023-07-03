import cv2
import numpy as np
import time

cap = cv2.VideoCapture('videos/object5.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

lr = 0

# Define dictionary to store object information
objects = {}

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
            # Check if object exists in the dictionary
            if index in objects:
                obj = objects[index]
                # Check if object size and center coordinate are within the error range
                if abs(width - obj[0]) / obj[0] <= 0.1 and \
                   abs(height - obj[1]) / obj[1] <= 0.1 and \
                   abs(centerX - obj[2]) / obj[2] <= 0.05 and \
                   abs(centerY - obj[3]) / obj[3] <= 0.05:
                    # Check if object has been maintained for 3 seconds
                    if current_time - obj[4] >= 3:
                        obj[5] = True  # Set flag for blue boundary
                else:
                    # Update object information
                    obj[:4] = [width, height, centerX, centerY]
                    obj[4] = current_time
                    obj[5] = False  # Reset flag for red boundary
            else:
                # Add new object to the dictionary
                objects[index] = [width, height, centerX, centerY, current_time, False]

            count_boxes += 1
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)

    for obj1 in objects.values():
        x1, y1, width1, height1, _, is_blue1 = obj1
        if is_blue1:
            continue  # Skip processing already absorbed objects

        for obj2 in objects.values():
            x2, y2, width2, height2, _, is_blue2 = obj2

            if is_blue2 or (obj1 is obj2):
                continue  # Skip already absorbed objects and self-comparison

            if x2 >= x1 and y2 >= y1 and (x2 + width2) <= (x1 + width1) and (y2 + height2) <= (y1 + height1):
                # obj2 is completely inside obj1, absorb obj2 into obj1
                obj1[5] = False  # Reset flag for red boundary
                obj2[5] = True  # Set flag for blue boundary
                break

    for obj in objects.values():
        x, y, width, height, _, is_blue = obj
        color = (255, 0, 0) if is_blue else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

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
