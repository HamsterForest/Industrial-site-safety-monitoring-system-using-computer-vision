import cv2 as cv

cap = cv.VideoCapture(0)

#라벨링 작업
def label_objects(frame):
    # Grayscale 변환
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 이진화
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    # 라벨링
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh)

    # 루프에서 라벨링 된 물체마다 사각형 그림
    for i in range(1, num_labels):
        if i<2:
            continue
        x, y, w, h, area = stats[i]
        if 10000>area>50:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return frame


while True:
    
    ret, frame = cap.read()

    if ret:
        labeled_frame = label_objects(frame)
        cv.imshow('Labeled Objects', labeled_frame)

    # q누르면, 나가기
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
