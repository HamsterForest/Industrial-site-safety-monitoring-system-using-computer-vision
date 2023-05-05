import cv2
import numpy as np
import time

# yolo 로드
net = cv2.dnn.readNet("weight_files_folder/cocodata/yolov3.weights", "weight_files_folder/cocodata/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 비디오 업로드
cap = cv2.VideoCapture('videos/vtest.avi')


classes = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/cocodata/coco.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes = [line.strip() for line in f.readlines()]

#마우스 드래그를 위한 것
isDragging = False
x0_m, y0_m, w_m, h_m = -1, -1, -1, -1

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
term=10 # term 조절 변수는 여기
initial_flag=True

#사용자 설정 모니터링 범위를 위해 프레임을 지정된 크기로 자른다.
user_flag = -1
def cut_frame(frame, pt1, pt2):
    
    x1, y1 = pt1
    x2, y2 = pt2
    
    return frame[y1:y2, x1:x2]

#사용자 마우스 드래그
def onMouse(event, x, y, flags, param):
    global isDragging, x0_m, y0_m, w_m, h_m, user_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        user_flag=1
        isDragging = True
        x0_m = x
        y0_m = y
        w_m = 0
        h_m = 0
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            w_m = x - x0_m
            h_m = y - y0_m
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w_m = x - x0_m
            h_m = y - y0_m

def yolo(frame, pt1, pt2):
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

def drawing(frame, idxs, boxes, term, pt1, pt2): 
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
    return frame

while True:
    #사용자 지정 범위의 변수선언
    pt1=(x0_m,y0_m)
    pt2=(x0_m+w_m,y0_m+h_m)
    
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    #사용자 지정 범위 
    cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', onMouse)

    if w_m>0 and h_m>0 and user_flag==1:
        if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
            cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
        if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
            w_m=0
            h_m=0
        
   
    #yolo term 조절
    if initial_flag==True:
        prev_time = time.time()
        if isDragging==False or( isDragging==True and (w_m<0 and h_m<0)):# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
            if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                idxs, boxes  = yolo(frame, pt1, pt2)
            else:
                idxs, boxes  = yolo(frame, (0,0), (640,480))
        initial_flag=False

    #사람수 표시는 항상 그린다.
    #사람수 text배경 -검은색
    cv2.rectangle(frame, org_back, org_back2, (0, 0, 0), -1)
    cv2.putText(frame, 'People : {}'.format(len(idxs)), org, font, 
                font_scale, color, thickness, cv2.LINE_AA)
    

    lapsed_time = time.time() - prev_time
    if lapsed_time > (1./ term):
        initial_flag=True

    #사람들 사각형 바운더리 그리기
    if isDragging == False or( isDragging==True and (w_m<0 and h_m<0)):# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
        if user_flag == 1 and w_m>100 and h_m>100:# 사각형이 너무 작으면, 전체 모니터링 수행
            frame = drawing(frame, idxs, boxes, term, pt1, pt2)
        else:
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480))
    
    cv2.imshow('frame', frame)

    keycode=cv2.waitKey(25)
    if keycode == ord('q'):
        break
    elif keycode == ord('u'):# 사용자 지정 범위 끄기
        user_flag=-1

cap.release()
cv2.destroyAllWindows()
