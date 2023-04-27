import cv2
import numpy as np
import time

# yolo 로드
net = cv2.dnn.readNet("weight_files_folder/yolov3_1/yolov3.weights", "weight_files_folder/yolov3_1/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 업로드
cap = cv2.VideoCapture('videos/vtest.avi')


classes = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/yolov3_1/coco.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes = [line.strip() for line in f.readlines()]

#영상에 글자를 넣기 위한 사전 설정
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50) # 글자 시작지점 글자의 왼쪽 하단
org_back = (org[0]-20, org[1]-25) # 글자 배경을 위한 사각형 시작 좌표
org_back2 = (org[0]+200, org[1]+5)
font_scale = 1
color = (255, 255, 255)
thickness = 2

#yolo term 조절
prev_time=0
term=15 # term 조절 변수는 여기
initial_flag=True

#모니터링 좌표 지정 비디오 사이즈는 640, 480 으로 고정 
pt1 = (50, 50)
pt2 = (400, 300)

#사용자 설정 모니터링 범위를 위해 프레임을 지정된 크기로 자른다.
def cut_frame(frame, pt1, pt2):
    
    x1, y1 = pt1
    x2, y2 = pt2
    
    return frame[y1:y2, x1:x2]

def yolo(frame):
    #지정된 사이즈로 프레임 자르기
    frame = cut_frame(frame, pt1, pt2)
    
    # 이미지를 그대로 넣는 것이 아니라, blob으로 넣게 된다.
    # blob은 이미지의 픽셀정보와 크기정보, 색의 채널 정보들을 가지는 행렬의 형태이다.
    # blop의 사이즈가 클수록 accuracy가 높아지지만 연산 시간이 늘어나게 된다.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(320, 320), 
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    #노이즈 제거를 위해 선언함 boxes에는 상자의 위치 좌표가 표시되고 confidences는 각 boxes에 대한 confidence
    confidences=[]
    boxes=[]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if classes[class_id] == "person" and confidence > 0.8: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return idxs, boxes

def drawing(frame, idxs, boxes, term): 
    # 그릴때, 잘려진 사진을 이용해 yolo를 했음을 고려하여, 좌표를 조정해야 한다.
    # 사람 수 세기
    people_count=len(idxs)


    if people_count>0:
        for i in idxs.flatten():
            box=boxes[i]
            left=box[0]
            top=box[1]
            w=box[2]
            h=box[3]
            if term>8:
                cv2.rectangle(frame, (left+pt1[0], top+pt1[1]), (left+w+pt1[0], top+h+pt1[1]), (0, 255, 0), 2)
    
    #모니터링 범위는 사각형으로 표시 된다.
    cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 2)

    #사람수 text배경 -검은색
    cv2.rectangle(frame, org_back, org_back2, (0, 0, 0), -1)
    #사람수 text - 흰색
    cv2.putText(frame, 'People : {}'.format(people_count), org, font, 
                font_scale, color, thickness, cv2.LINE_AA)

    
    return frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    #yolo term 조절
    if initial_flag==True:
        prev_time = time.time()
        idxs, boxes  = yolo(frame)
        initial_flag=False

    lapsed_time = time.time() - prev_time
    if lapsed_time > (1./ term):
        initial_flag=True

    frame = drawing(frame, idxs, boxes, term)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
