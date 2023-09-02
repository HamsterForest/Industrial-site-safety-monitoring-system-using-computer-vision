import cv2
import numpy as np
import time
import tkinter as tk #인터페이스

#----------------------------------------------------------------------------------------------------#

# yolo 로드=========================================
net = cv2.dnn.readNet("weight_files_folder/cocodata/yolov3.weights", "weight_files_folder/cocodata/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/cocodata/coco.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes = [line.strip() for line in f.readlines()]
# yolo 로드=========================================

# yolo 라바콘 로드 =================================
net_t = cv2.dnn.readNet("weight_files_folder/safety/yolov3.weights", "weight_files_folder/safety/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names_t = net_t.getLayerNames()
output_layers_t = [layer_names_t[i - 1] for i in net_t.getUnconnectedOutLayers()]

classes_t = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/safety/obj.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes_t = [line.strip() for line in f.readlines()]
# yolo 라바콘 로드 =================================

# yolo 헬멧 로드 =================================
net_h = cv2.dnn.readNet("weight_files_folder/helmet/yolov3.weights", "weight_files_folder/helmet/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names_h = net_h.getLayerNames()
output_layers_h = [layer_names_h[i - 1] for i in net_h.getUnconnectedOutLayers()]

classes_h = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/helmet/obj.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes_h = [line.strip() for line in f.readlines()]

# yolo 헬멧 로드 =================================

#마우스 드래그를 위한 것
isDragging = False
x0_m, y0_m, w_m, h_m = -1, -1, -1, -1

#자동범위조정을 위한 것
auto_boundary_tops=[]
auto_boundary_bottoms=[]
auto_boundary_lefts=[]
auto_boundary_rights=[]
#영상에 글자를 넣기 위한 사전 설정 
font = cv2.FONT_HERSHEY_SIMPLEX
#영상에 글자를 넣기 위한 사전 설정 - 사람수 count
count_cor = (20, 20) # 글자 시작지점 글자의 왼쪽 하단
count_cor_back = (count_cor[0]-10, count_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
count_cor_back2 = (count_cor[0]+200, count_cor[1]+5)
count_font_scale = 0.5
count_color = (255, 255, 255)
count_thickness = 1
#영상에 글자를 넣기 위한 사전 설정 - 안전모 모니터링 경고 
helmet_caut_cor = (100, 70) # 글자 시작지점 글자의 왼쪽 하단
helmet_caut_cor_back = (helmet_caut_cor[0]-10, helmet_caut_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
helmet_caut_cor_back2 = (helmet_caut_cor[0]+200, helmet_caut_cor[1]+5)
helmet_caut_font_scale = 0.5
helmet_caut_color = (255, 255, 255)
helmet_caut_thickness = 1
#영상에 글자를 넣기 위한 사전 설정 - 접근금지 모니터링
offlimit_cor = (20, 20) # 글자 시작지점 글자의 왼쪽 하단
offlimit_cor_back = (offlimit_cor[0]-10, offlimit_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
offlimit_cor_back2 = (offlimit_cor[0]+260, offlimit_cor[1]+5)
offlimit_font_scale = 0.5
offlimit_color = (255, 255, 255)
offlimit_thickness = 1
#영상에 글자를 넣기 위한 사전 설정 - 작업인원 할당 - 경고 출력
count_caut_cor = (100, 430)
count_caut_cor_back = (count_caut_cor[0]-10, count_caut_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
count_caut_cor_back2 = (count_caut_cor[0]+205, count_caut_cor[1]+5)
count_caut_font_scale = 0.5
count_caut_color = (255, 255, 255)
count_caut_thickness = 1
#영상에 글자를 넣기 위한 사전 설정 - 접근금지구역 모니터링 - 경고 출력
offlimit_caut_cor = (100, 430)
offlimit_caut_cor_back = (offlimit_caut_cor[0]-10, offlimit_caut_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
offlimit_caut_cor_back2 = (offlimit_caut_cor[0]+300, offlimit_caut_cor[1]+5)
offlimit_caut_cor_b = (100, 380)
offlimit_caut_cor_back_b = (offlimit_caut_cor_b[0]-10, offlimit_caut_cor_b[1]-15) # 글자 배경을 위한 사각형 시작 좌표
offlimit_caut_cor_back2_b = (offlimit_caut_cor_b[0]+300, offlimit_caut_cor_b[1]+5)
offlimit_caut_cor_c = (100, 330)
offlimit_caut_cor_back_c = (offlimit_caut_cor_c[0]-10, offlimit_caut_cor_c[1]-15) # 글자 배경을 위한 사각형 시작 좌표
offlimit_caut_cor_back2_c = (offlimit_caut_cor_c[0]+300, offlimit_caut_cor_c[1]+5)
offlimit_caut_font_scale = 0.5
offlimit_caut_color = (255, 255, 255)
offlimit_caut_thickness = 1
#yolo term 조절
prev_time=0
initial_flag=True

#사용자 설정 모니터링 범위를 위해 프레임을 지정된 크기로 자른다.
user_flag = -1

#-----------------------------------------------------------------------------------------------------#
#cut_frame
#onMouse
#helmet_yolo
#yolo
#drawing
#draw_caution
#auto_boundary
#auto_range_filter
#stockpiled_monitoring_core
#stockpiled_monitoring
#offlimit_monitoring
#workers_counts_monitoring
#helmet_monitoring
#main_loop
#bck_btn
#first_btn
#first_btn2
#second_btn
#second_btn2
#third_btn
#third_btn2
#forth_btn
#forth_btn2

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

#헬멧 감지 욜로
def helmet_yolo(frame, pt1, pt2, mod):
    #지정된 사이즈로 프레임 자르기
    if mod==0:#지정된 범위 안에만 감지
        frame = cut_frame(frame, pt1, pt2)
    else:
        copyfrmae=frame.copy()
        copyframe=cv2.rectangle(copyfrmae, pt1, pt2, (0,0,0), -1)
        frame=copyframe.copy()
    
    # 이미지를 그대로 넣는 것이 아니라, blob으로 넣게 된다.
    # blob은 이미지의 픽셀정보와 크기정보, 색의 채널 정보들을 가지는 행렬의 형태이다.
    # blop의 사이즈가 클수록 accuracy가 높아지지만 연산 시간이 늘어나게 된다.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), 
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net_h.setInput(blob)
    outputs = net_h.forward(output_layers_h)

    #노이즈 제거를 위해 선언함 boxes에는 상자의 위치 좌표가 표시되고 confidences는 각 boxes에 대한 confidence
    confidences=[]
    boxes=[]
    names=[]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                        frame.shape[1], frame.shape[0]])
            left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
            confidences.append(float(confidence))
            boxes.append([left,top,w,h])
            names.append(classes_h[class_id])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.6)
    
    return idxs, boxes, names

#사람 감지 욜로
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
            
            if classes[class_id] == "person" and confidence > 0.5: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    
    return idxs, boxes

#사람에 바운더리 사각형 그리기
def drawing(frame, idxs, boxes, term, pt1, pt2, names_check, names): 
    
    if names_check==0:
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
                else:
                    cv2.circle(frame, (left+int(w/2)+pt1[0], top+int(h/2)+pt1[1]), 1, (0, 0, 255), 2)
        return frame
    elif names_check==1:#헬멧 사각형 그리기
        nohelmet_count=0
        if len(idxs)>0:
            for i in idxs.flatten():
                box=boxes[i]
                left=box[0]
                top=box[1]
                w=box[2]
                h=box[3]
                if names[i]=='head':
                    cv2.rectangle(frame, (left+pt1[0], top+pt1[1]), (left+w+pt1[0], top+h+pt1[1]), (0, 0, 255), 2)
                    cv2.putText(frame, 'No Hard-hat',(left+pt1[0], top+pt1[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    nohelmet_count+=1
                else:
                    cv2.rectangle(frame, (left+pt1[0], top+pt1[1]), (left+w+pt1[0], top+h+pt1[1]), (0, 255, 0), 2)
                    cv2.putText(frame, 'Hard-hat',(left+pt1[0], top+pt1[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        return frame, nohelmet_count



#경고 띄우기
def draw_caution(frame, idxs, type):
    if type==1:#접근금지 모니터링
        cv2.rectangle(frame, offlimit_caut_cor_back, offlimit_caut_cor_back2, (0, 0, 0), -1)
        cv2.circle(frame,(offlimit_caut_cor[0]-45,offlimit_caut_cor[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : people in no access area'.format(len(idxs)), offlimit_caut_cor, font, 
                    offlimit_caut_font_scale, offlimit_caut_color, offlimit_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==2:#작업인원수 모니터링
        cv2.rectangle(frame, count_caut_cor_back, count_caut_cor_back2, (0, 0, 0), -1)
        cv2.circle(frame,(count_caut_cor[0]-45,count_caut_cor[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : lack of workers'.format(len(idxs)), count_caut_cor, font, 
                    count_caut_font_scale, count_caut_color, count_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==3:#적치물제한구역 모니터링
        cv2.rectangle(frame, count_caut_cor_back, (count_caut_cor[0]+225, count_caut_cor[1]+5), (0, 0, 0), -1)
        cv2.circle(frame,(count_caut_cor[0]-45,count_caut_cor[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : Stockpiles in area', count_caut_cor, font, 
                    count_caut_font_scale, count_caut_color, count_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==4:#안전모 착용 모니터링
        cv2.rectangle(frame, count_caut_cor_back, count_caut_cor_back2, (0, 0, 0), -1)
        cv2.circle(frame,(count_caut_cor[0]-45,count_caut_cor[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : No Hard-hat', count_caut_cor, font, 
                    count_caut_font_scale, count_caut_color, count_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==5:#접근금지랑 적치물제한 동시에 경고 송출시 겹치는 것을 방지 하기 위함-접근금지경고
        cv2.rectangle(frame, offlimit_caut_cor_back_b, offlimit_caut_cor_back2_b, (0, 0, 0), -1)
        cv2.circle(frame,(offlimit_caut_cor_b[0]-45,offlimit_caut_cor_b[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : people in no access area', offlimit_caut_cor_b, font, 
                    offlimit_caut_font_scale, offlimit_caut_color, offlimit_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==6:#작업인원수 경고와 적치물제한 동시에 경고 송출시 겹치는 것을 방지하기 위함-적치물경고
        cv2.rectangle(frame, offlimit_caut_cor_back_b, offlimit_caut_cor_back2_b, (0, 0, 0), -1)
        cv2.circle(frame,(offlimit_caut_cor_b[0]-45,offlimit_caut_cor_b[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : Stockpiles in area', offlimit_caut_cor_b, font, 
                    offlimit_caut_font_scale, offlimit_caut_color, offlimit_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==7:#작업인원수, 적치물제한, 접근금지 동사에 경고 송출시 겹치는 것을 방지하기 위함-접근금지
        cv2.rectangle(frame, offlimit_caut_cor_back_c, offlimit_caut_cor_back2_c, (0, 0, 0), -1)
        cv2.circle(frame,(offlimit_caut_cor_c[0]-45,offlimit_caut_cor_c[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : people in no access area', offlimit_caut_cor_c, font, 
                    offlimit_caut_font_scale, offlimit_caut_color, offlimit_caut_thickness, cv2.LINE_AA)
        return frame
    elif type==8:#안전모 경고문
        cv2.rectangle(frame, helmet_caut_cor_back, helmet_caut_cor_back2, (0, 0, 0), -1)
        cv2.circle(frame,(helmet_caut_cor[0]-45,helmet_caut_cor[1]-5),15,(0,0,255),-1)
        cv2.putText(frame, 'Caution : No Hard-hat', helmet_caut_cor, font, 
                    helmet_caut_font_scale, helmet_caut_color, helmet_caut_thickness, cv2.LINE_AA)
        return frame


#범위자동설정
def auto_boundary(frame):
    #10초동안 실행할것
    #yolo term 기능과 연동되게 할것
    #바운더리사각형을 반환할것

    global auto_boundary_tops, auto_boundary_bottoms, auto_boundary_lefts, auto_boundary_rights

    # 이미지를 그대로 넣는 것이 아니라, blob으로 넣게 된다.
    # blob은 이미지의 픽셀정보와 크기정보, 색의 채널 정보들을 가지는 행렬의 형태이다.
    # blop의 사이즈가 클수록 accuracy가 높아지지만 연산 시간이 늘어나게 된다.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), 
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net_t.setInput(blob)
    outputs = net_t.forward(output_layers_t)

    #노이즈 제거를 위해 선언함 boxes에는 상자의 위치 좌표가 표시되고 confidences는 각 boxes에 대한 confidence
    confidences=[]
    boxes=[]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if classes_t[class_id] == "Safety Cone" and confidence > 0.5: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    if len(idxs)>0:
        for i in idxs.flatten():#좌표들 저장용 전역변수 사용.
            box=boxes[i]
            auto_boundary_tops.append(box[1])
            auto_boundary_bottoms.append(box[1]+box[3])
            auto_boundary_lefts.append(box[0])
            auto_boundary_rights.append(box[0]+box[2])
    
    return idxs, boxes

#자동범위 조정시 범위 필터링용 함수
def auto_range_filter(numbers, mod, term):
    duplicates = [num for num in numbers if numbers.count(num) > int(term*10*0.2)]
    if mod==0:
        return min(duplicates)
    else:
        return max(duplicates)    

def stockpiled_monitoring_core(fgbg, current_time, objects, objects_duration, frame, pt1, pt2):
    copied_frame=frame.copy()
    frame=cut_frame(frame,pt1,pt2)
    #가우시안 블러링-엣지검출전 노이즈 제거
    blur=cv2.GaussianBlur(frame,(15,15),5)
    #전경마스크 얻기
    fgmask=fgbg.apply(blur,learningRate=0)
    #모폴로지-타원형 커널
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51))
    #모폴로지-클로징-흰색영역의 검은색 구멍을 메운다
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    blue_counts=0
    red_counts=0
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
                    if  abs(width - objects[index][0]) / objects[index][0] <= 0.4 and \
                        abs(height - objects[index][1]) / objects[index][1] <= 0.4 and \
                        abs(centerX - objects[index][2]) / objects[index][2] <= 0.2 and \
                        abs(centerY - objects[index][3]) / objects[index][3] <= 0.2:
                        # 3초이상 적치시 파란색으로 경계 바꾸어 그림 아니면 빨간색
                        if current_time - objects_duration[index] >= 3:
                            blue_counts+=1
                            cv2.rectangle(copied_frame, (x+pt1[0], y+pt1[1]), (x + width + pt1[0], y + height+pt1[1]), (255, 0, 0), 2)
                        else:
                            cv2.rectangle(copied_frame, (x+pt1[0], y+pt1[1]), (x + width + pt1[0], y + height+pt1[1]), (0, 0, 255), 2)
                    else:
                        # 오브젝트 사이즈와 중심점 좌표, 시간 업데이트(오차범위 안에 없으니 시간 초기화)
                        objects[index] = [width, height, centerX, centerY]
                        objects_duration[index] = current_time
                else:
                    # 새로운 오브젝트 추가
                    objects[index] = [width, height, centerX, centerY]
                    objects_duration[index] = current_time
                red_counts+=1
                red_counts= red_counts-blue_counts
                cv2.circle(copied_frame, (centerX+pt1[0], centerY+pt1[1]), 1, (0, 255, 0), 2)

    return copied_frame, objects, objects_duration,blue_counts,red_counts

#적치물제한구역모니터링
def stockpiled_monitoring(term, auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/object7.mp4')
    else:
        cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)
    lr = 0
    objects = {}
    objects_duration = {}
    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0

    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))
        
        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)
                

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            initial_flag=False

        #욜로 사람수 세기 메인
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
            if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                frame, objects, objects_duration, blue_counts,_=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame, objects, objects_duration, blue_counts,_=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
            else:
                frame, objects, objects_duration, blue_counts,_=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,(0,0),(640,480))

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, offlimit_cor_back, offlimit_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'storage restrictions monitoring', offlimit_cor, font, 
                        offlimit_font_scale, offlimit_color, offlimit_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,3)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

#적치물제한구역모니터링과 접근금지구역모니터링 합친것
def stockpiled_offlimit_monitoring(term, auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/object7.mp4')
    else:
        cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)
    lr = 0
    objects = {}
    objects_duration = {}
    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0

    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))
        
        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)
                

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            initial_flag=False

        #욜로 사람수 세기 메인
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
            if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
            else:
                frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,(0,0),(640,480))

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, (15,5), (350,30), (0, 0, 0), -1)
            cv2.putText(frame, 'storage restrictions, off-limit monitoring', offlimit_cor, font, 
                        offlimit_font_scale, offlimit_color, offlimit_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,3)

        if red_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,5)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

#접근금지구역모니터링
def off_limit_monitoring(term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/vtest5.mp4')
    else:
        cap = cv2.VideoCapture(0)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구

            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes  = yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes  = yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                else:
                    idxs, boxes  = yolo(frame, (0,0), (640,480))
                    people_count=len(idxs)
            initial_flag=False


        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, offlimit_cor_back, offlimit_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Off-limit monitoring. Count : {}'.format(people_count), offlimit_cor, font, 
                        offlimit_font_scale, offlimit_color, offlimit_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs,1)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            if  user_flag==1 and w_m>100 and h_m>100:# 사각형이 너무 작으면, 전체 모니터링 수행
                frame = drawing(frame, idxs, boxes, term, pt1, pt2,0,0)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame = drawing(frame, idxs, boxes, term, pt1, pt2,0,0)    
            else:
                frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 자동범위조정 진입
            pass
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

#작업인원수모니터링
def workers_counts_monitoring(num, term, auto_range,sample_video):#num은 할당 인원수
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/cone_people3.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구

            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes  = yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes  = yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                else:
                    idxs, boxes  = yolo(frame, (0,0), (640,480))
                    people_count=len(idxs)
            initial_flag=False


        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            if  user_flag==1 and w_m>100 and h_m>100:# 사각형이 너무 작으면, 전체 모니터링 수행
                frame = drawing(frame, idxs, boxes, term, pt1, pt2,0,0)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame = drawing(frame, idxs, boxes, term, pt1, pt2,0,0)    
            else:
                frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 자동범위조정 진입
            pass
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

#안전모착용모니터링
def helmet_monitoring(term, auto_range, sample_video):#num은 할당 인원수
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/safety.mp4')
    else:
        cap = cv2.VideoCapture(0)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    nohelmet_count=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구

            #욜로 헬멧 감지
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, names  = helmet_yolo(frame, pt1, pt2,0)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, names  = helmet_yolo(frame, pt1, pt2,0)
                else:
                    idxs, boxes, names  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Hard-hat Monitoring', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #헬멧 확인 경고
        if len(idxs)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs,4)
        
        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
        
        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            if  user_flag==1 and w_m>100 and h_m>100:# 사각형이 너무 작으면, 전체 모니터링 수행
                frame, nohelmet_count = drawing(frame, idxs, boxes, term, pt1, pt2,1,names)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame, nohelmet_count = drawing(frame, idxs, boxes, term, pt1, pt2,1,names)    
            else:
                frame, nohelmet_count = drawing(frame, idxs, boxes, term, (0,0), (640,480),1,names)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 자동범위조정 진입
            pass
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def cut_yolo(frame, pt1, pt2):#지정한 영역은 욜로를 수행하지 않는다.
    #지정된 사이즈로 프레임 자르기
    copyfrmae=frame.copy()
    copyframe=cv2.rectangle(copyfrmae, pt1, pt2, (0,0,0), -1)
    
    # 이미지를 그대로 넣는 것이 아니라, blob으로 넣게 된다.
    # blob은 이미지의 픽셀정보와 크기정보, 색의 채널 정보들을 가지는 행렬의 형태이다.
    # blop의 사이즈가 클수록 accuracy가 높아지지만 연산 시간이 늘어나게 된다.
    blob = cv2.dnn.blobFromImage(copyframe, scalefactor=1/255, size=(320, 320), 
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
            
            if classes[class_id] == "person" and confidence > 0.5: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    
    return idxs, boxes

#지정된 영역 안에 사람이 있으면 offlimit=1 반환, 지정된 영역안에 사람수를 제외하고 사람수를 반환한다.
def yolo_seperate(frame, pt1, pt2):#지정한 영역 안에서의 인원을 따로 센다.
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
    offlimit=0
    offlimit_count=0
    people_count=0

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if classes[class_id] == "person" and confidence > 0.5: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    #전체범위모니터링에서는 count workers만
    if pt1!=(0,0) and pt2!=(640,480):
        for i in idxs.flatten():
            box=boxes[i]
            left=box[0]
            top=box[1]
            w=box[2]
            h=box[3]
            #사람의 바운더리 사각형이 일부라도 지정한 범위를 침범하는 경우 경고
            if (pt2[0]>left+w>pt1[0] or pt2[0]>left>pt1[0]) and (pt2[1]>(top+h)>pt1[1] or pt2[1]>top>pt1[1]):
                offlimit=1
                offlimit_count+=1
            

    people_count=len(idxs)-offlimit_count
    return idxs, boxes, offlimit, people_count

def cut_drawing(frame, idxs, boxes, term, pt1, pt2, names_check, names): 
    
    if names_check==0:
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
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
        return frame
    elif names_check==1:
        nohelmet_count=0
        if len(idxs)>0:
            for i in idxs.flatten():
                box=boxes[i]
                left=box[0]
                top=box[1]
                w=box[2]
                h=box[3]
                
                if names[i]=='head':
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 0, 255), 2)
                    cv2.putText(frame, 'No Hard-hat',(left, top),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                    nohelmet_count+=1
                else:
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Hard-hat',(left, top),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        return frame, nohelmet_count

def workers_counts_stockpiles_monitoring(num,term,auto_range,sample_video):
    
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/cone_box.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1

            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes  = cut_yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes  = cut_yolo(frame, pt1, pt2)
                    people_count=len(idxs)
                else:
                    idxs, boxes  = cut_yolo(frame, (0,0), (0,0))#전체 욜로만 시행
                    people_count=len(idxs)
            initial_flag=False

        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            if  user_flag==1 and w_m>100 and h_m>100:# 사각형이 너무 작으면, 전체 모니터링 수행
                frame = cut_drawing(frame, idxs, boxes, term, pt1, pt2,0,0)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame = cut_drawing(frame, idxs, boxes, term, pt1, pt2,0,0)    
            else:
                frame = cut_drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_off_limit_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/cone_people4.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구

            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False


        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)
        
        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,5)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 자동범위조정 진입
            pass
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_off_limit_stockpiled_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/cone_box.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
        
        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)
        
        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,7)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True

        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_off_limit_stockpiled_helmet_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)
        
        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,7)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_off_limit_helmet_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)
        
        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,7)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_stockpiled_helmet_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)

        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def off_limit_stockpiled_helmet_monitoring(term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0
    
    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        if auto_range==1:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,7)

        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def off_limit_helmet_monitoring(term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, pt1, pt2)
                else:
                    idxs, boxes, offlimit, people_count  = yolo_seperate(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False

        if auto_range==1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #접근금지 확인 및 경고 표현
        if offlimit==1:
            frame=draw_caution(frame, idxs,7)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def stockpiled_helmet_monitoring(term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,1)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False
        
        #적치물제한 감시 부분 - 텀 조절 없이 실시간으로 하도록 함
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    frame, objects, objects_duration, blue_counts,red_counts=stockpiled_monitoring_core(fgbg,current_time,objects,objects_duration,frame,pt1,pt2)
                    people_count=len(idxs)

        #적치물 확인 및 경고
        if blue_counts>0 and auto_range!=1:
            frame=draw_caution(frame, 0,6)

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range==1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)

        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, (0,0), (640,480),0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, (0,0), (640,480),1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

def workers_counts_helmet_monitoring(num,term,auto_range,sample_video):
    # 비디오 업로드
    if sample_video==1:
        cap = cv2.VideoCapture('videos/allforone.mp4')
    else:
        cap = cv2.VideoCapture(0)
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

    root.withdraw()#인터페이스 숨기기
    #isDragging => 마우스를 드래그 중인가
    #x0_m, y0_m, w_m, h_m => 마우스로 그린 사각형 좌표
    #prev_time => yolo term 관련
    #initial_flag => initial_flag가 1일 때만 욜로를 진행한다. yolo term과 관련됨.
    #usef_flag => 1이면 사용자지정범위가 켜진다.
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag
    people_count=0
    offlimit=0
    auto_range_on=0#auto_range하고 있는 중인가?
    auto_range_on_count=0#auto_range하고 얼마나 욜로에 진입했나?
    blue_counts=0
    red_counts=0
    objects = {}
    objects_duration = {}
    first_boundary_made=0#자동혹은 수동 범위 확정에서 최초로 경계를 그릴때, fgbg를 초기화 하기 위한 변수
    switch=-1#한번은 coco 한번은 helmet 번갈아가면서 yolo 하기 위한 변수
    idxs=[]
    idxs_h=[]
    boxes=[]
    boxes_h=[]
    names_h=[]
    nohelmet_count=0

    while True:
        #사용자 지정 범위의 변수선언
        pt1=(x0_m,y0_m)
        pt2=(x0_m+w_m,y0_m+h_m)
        
        #비디오캡쳐
        ret, frame = cap.read()
        #비디오 없으면 종료
        if not ret:
            break
        #비디오 사이즈 재조정
        frame = cv2.resize(frame, (640, 480))

        current_time = time.time()

        #자동범위조정 설정시 사용자 지정범위 끄기=> 나중에 다시 켜야함.
        if auto_range==1:
            auto_range_on=1
            user_flag=-1#사용자지정범위 끄기

        #사용자 지정 범위 
        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', onMouse)
        #모니터링 바운더리 사각형 그려지는 곳-욜로수행범위를 실질적으로 제한하는 것은 pt1 pt2의 변경
        if w_m>0 and h_m>0 and user_flag==1:
            if w_m<100 or h_m<100: # 작은 사각형은 다른 색 사용
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
                first_boundary_made=1
            if isDragging==False and (w_m<100 or h_m<100): # 최종적으로 너무 작은 상자가 그려지면 지워짐
                first_boundary_made=0
                w_m=0
                h_m=0
        
        #자동범위 확정적 그리기
        if auto_range_on!=1 and user_flag!=1:#자동범위 실행중에는 그리지 않음
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 3)
            if first_boundary_made==1:#경계가 최초로 확정된 때에는 fgbg를 초기화
                first_boundary_made=0
                fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=250, detectShadows=False)

        #yolo term 조절, yolo 진입, auto_range도 여기서 진입
        if initial_flag==True:
            prev_time = time.time()
            
            #자동 범위 조정 메인
            if auto_range==1 and auto_range_on==1:
                idxs, boxes=auto_boundary(frame)
                auto_range_on_count+=1
                if auto_range_on_count==(term*10):#10초동안 진행
                    auto_range_on=0#auto range종료
                    auto_range=0
                    #pt1과 pt2를 지정하면, 그곳에서만 욜로 수행, 그를 위해서 x0_m,y0_m,w_m,h_m 조정
                  
                    if len(auto_boundary_tops)>=6:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    elif 0<len(auto_boundary_tops)<6:
                        x0_m=min(auto_boundary_lefts)
                        w_m=max(auto_boundary_rights)-min(auto_boundary_lefts)
                        y0_m=0
                        h_m=480
                        pt1=(x0_m,y0_m)
                        pt2=(x0_m+w_m,y0_m+h_m)
                    else:
                        pass # 자동범위 조정 실패 문구
                    first_boundary_made=1
            
            #욜로 사람수 세기 메인
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==-1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs, boxes  = yolo(frame, pt1, pt2)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs, boxes  = yolo(frame, pt1, pt2)
                else:
                    idxs, boxes  = yolo(frame, (0,0), (640,480))
            initial_flag=False
            #헬멧 모니터링
            if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1 and switch==1:# 드래그 하는 동안에 그리고 사각형이 그려지는 동안에는 욜로하지 않음
                if user_flag == 1 and w_m>100 and h_m>100:#사용자 범위 설정시 욜로 범위가 너무 작으면 안된다.
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,0)
                elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, pt1, pt2,0)
                else:
                    idxs_h, boxes_h, names_h  = helmet_yolo(frame, (0,0), (640,480),0)
            initial_flag=False

        #사람수 표시는 항상 그린다.
        #사람수 text배경 -검은색
        if auto_range!=1:
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Allocated : {} Count : {}'.format(num,people_count), count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        else:#자동 범위 설정중일 경우
            cv2.rectangle(frame, count_cor_back, count_cor_back2, (0, 0, 0), -1)
            cv2.putText(frame, 'Detecting Traffic cone', count_cor, font, 
                        count_font_scale, count_color, count_thickness, cv2.LINE_AA)
        
        #사람수 확인 및 경고 표현
        if people_count<num and auto_range!=1:
            frame=draw_caution(frame, idxs,2)

        #헬멧 확인 경고
        if len(idxs_h)>0 and nohelmet_count>0 and auto_range!=1:
            frame=draw_caution(frame, idxs_h,8)

        #YOLO텀 조절용2
        lapsed_time = time.time() - prev_time
        if lapsed_time > (1./ term):
            initial_flag=True
            switch=switch*(-1)


        #자동범위에서 라바콘 그리기
        if auto_range==1:
            if len(idxs)>0:
                for i in idxs.flatten():
                    box=boxes[i]
                    left=box[0]
                    top=box[1]
                    w=box[2]
                    h=box[3]
                    cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

        #사람들 사각형 바운더리 그리기, 항상 전체모니터링 함.
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame = drawing(frame, idxs, boxes, term, pt1, pt2,0,0)
        
        #헬멧 사각형 바운더리 그리기
        if (isDragging==False or( isDragging==True and (w_m<0 and h_m<0))) and auto_range!=1:# 드래그 하는 동안에그리고 사각형이 그려지는 동안에 사람 바운더리 그리지 않음
            frame, nohelmet_count = drawing(frame, idxs_h, boxes_h, term, pt1, pt2,1,names_h)

        cv2.imshow('frame', frame)

        keycode=cv2.waitKey(25)
        if keycode == 27:#esc키 종료
            break
        elif keycode == ord('u'):# 사용자 지정 범위 끄기
            user_flag=-1
        elif keycode == ord('r'):# 디폴트화면 리셋
            fgbg=cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=250,detectShadows=False)
    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()#인터페이스 다시 등장

#메인루프
def main_loop(toggle1, toggle2, toggle3, toggle4,allocated,term,auto_range,sample_video):
    global isDragging, x0_m, y0_m, w_m, h_m, prev_time, initial_flag, user_flag,auto_boundary_tops, auto_boundary_bottoms, auto_boundary_lefts, auto_boundary_rights
    #전역변수 초기화
    isDragging = False
    x0_m, y0_m, w_m, h_m = -1, -1, -1, -1
    prev_time=0
    initial_flag=True
    user_flag = -1
    auto_boundary_tops=[]
    auto_boundary_bottoms=[]
    auto_boundary_lefts=[]
    auto_boundary_rights=[]

    root.withdraw()#인터페이스 숨기기

    if toggle1==1 and toggle2==0 and toggle3==0 and toggle4==0:#접근금지 모니터링
        off_limit_monitoring(term,auto_range,sample_video)
    elif toggle1==0 and toggle2==1 and toggle3==0 and toggle4==0:#작업인원수 모니터링
        workers_counts_monitoring(allocated,term,auto_range,sample_video)#할당인원, term 변수, auto_range여부
    elif toggle1==0 and toggle2==0 and toggle3==1 and toggle4==0:#적치물제한구역 모니터링
        stockpiled_monitoring(term, auto_range,sample_video)
    elif toggle1==0 and toggle2==0 and toggle3==0 and toggle4==1:#안전모착용 모니터링
        helmet_monitoring(term,auto_range,sample_video)
    elif toggle1==1 and toggle2==0 and toggle3==1 and toggle4==0:#접근금지+적치물제한
        stockpiled_offlimit_monitoring(term, auto_range,sample_video)
    elif toggle1==0 and toggle2==1 and toggle3==1 and toggle4==0:#적치물제한+작업인원수제한
        workers_counts_stockpiles_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==1 and toggle2==1 and toggle3==0 and toggle4==0:#접근금지+작업인원수제한
        workers_counts_off_limit_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==1 and toggle2==1 and toggle3==1 and toggle4==0:#적치물제한+접근금지+작업인원수제한
        workers_counts_off_limit_stockpiled_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==1 and toggle2==1 and toggle3==1 and toggle4==1:#적치물제한+접근금지+작업인원수제한+안전모 
        workers_counts_off_limit_stockpiled_helmet_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==1 and toggle2==1 and toggle3==0 and toggle4==1:#접근금지+작업인원수제한+안전모 모니터링
        workers_counts_off_limit_helmet_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==0 and toggle2==1 and toggle3==1 and toggle4==1:#적치물제한+작업인원수제한+안전모 모니터링
        workers_counts_stockpiled_helmet_monitoring(allocated,term,auto_range,sample_video)
    elif toggle1==1 and toggle2==0 and toggle3==1 and toggle4==1:#적치물제한+접근금지+안전모 모니터링
        off_limit_stockpiled_helmet_monitoring(term,auto_range,sample_video)
    elif toggle1==1 and toggle2==0 and toggle3==0 and toggle4==1:#접근금지+안전모 모니터링
        off_limit_helmet_monitoring(term,auto_range,sample_video)
    elif toggle1==0 and toggle2==0 and toggle3==1 and toggle4==1:#적치물제한+안전모 모니터링
        stockpiled_helmet_monitoring(term,auto_range,sample_video)
    elif toggle1==0 and toggle2==1 and toggle3==0 and toggle4==1:#작업인원수+안전모 모니터링
        workers_counts_helmet_monitoring(allocated,term,auto_range,sample_video)

    root.deiconify()#인터페이스 다시 등장
    

#인터페이스 정의
root = tk.Tk()
root.title("Monitoring system")
root.geometry("640x400")
root.resizable(False, False)

tk.Label(root,text='[테스트용 영상을 사용해서 실행]').place(x=380, y=355)
tk.Label(root, text="ON").place(x=580, y=340)
toggle_var1 = tk.IntVar()
toggle_var1.set(1)
toggle_check1 = tk.Checkbutton(root,variable=toggle_var1)
toggle_check1.place(x=580, y=355)

tk.Label(root,text='[주요모니터링 기능]').place(x=50, y=10)

tk.Label(root,text='1. 접근금지구역 모니터링').place(x=50, y=40)
toggle_var2 = tk.IntVar()
toggle_var2.set(0)
toggle_check2 = tk.Checkbutton(root,variable=toggle_var2)
toggle_check2.place(x=210, y=38)
tk.Label(root,text='2. 작업인원수 모니터링').place(x=50, y=70)
toggle_var3 = tk.IntVar()
toggle_var3.set(0)
toggle_check3 = tk.Checkbutton(root,variable=toggle_var3)
toggle_check3.place(x=210, y=68)
tk.Label(root,text='할당인원 수 설정 : ').place(x=250, y=70)
scale1 = tk.Scale(root, from_=2, to=10,orient="horizontal", length=200)
scale1.place(x=350, y=50,)
tk.Label(root,text='3. 적치물제한구역 모니터링').place(x=50, y=100)
toggle_var4 = tk.IntVar()
toggle_var4.set(0)
toggle_check4 = tk.Checkbutton(root,variable=toggle_var4)
toggle_check4.place(x=210, y=98)
tk.Label(root,text='4. 안전모착용 모니터링').place(x=50, y=130)
toggle_var5 = tk.IntVar()
toggle_var5.set(0)
toggle_check5 = tk.Checkbutton(root,variable=toggle_var5)
toggle_check5.place(x=210, y=128)


tk.Label(root,text='[부가 기능 설정]').place(x=50, y=168)
tk.Label(root,text='자동범위설정 기능 사용').place(x=50, y=190)
toggle_var6 = tk.IntVar()
toggle_var6.set(0)
toggle_check6 = tk.Checkbutton(root,variable=toggle_var6)
toggle_check6.place(x=210, y=188)
tk.Label(root,text='모니터링 빈도 : ').place(x=50, y=230)
scale2 = tk.Scale(root, from_=1, to=60,orient="horizontal", length=300)
scale2.set(3)
scale2.place(x=160, y=210,)

button1 = tk.Button(root, text="모니터링 시작",command=lambda : main_loop(toggle_var2.get(),toggle_var3.get(),toggle_var4.get(),toggle_var5.get(),scale1.get(),scale2.get(),toggle_var6.get(),toggle_var1.get()))
button1.place(x=50, y=320, width=250, height=50)

root.mainloop()#인터페이스 실행


