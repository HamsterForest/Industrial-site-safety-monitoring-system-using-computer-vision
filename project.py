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
#영상에 글자를 넣기 위한 사전 설정 - 경고 출력
caut_cor = (100, 430)
caut_cor_back = (caut_cor[0]-10, caut_cor[1]-15) # 글자 배경을 위한 사각형 시작 좌표
caut_cor_back2 = (caut_cor[0]+205, caut_cor[1]+5)
caut_font_scale = 0.5
caut_color = (255, 255, 255)
caut_thickness = 1
#yolo term 조절
prev_time=0
initial_flag=True

#사용자 설정 모니터링 범위를 위해 프레임을 지정된 크기로 자른다.
user_flag = -1

#-----------------------------------------------------------------------------------------------------#
#cut_frame
#onMouse
#yolo
#drawing
#draw_caution
#auto_boundary
#workers_counts_monitoring
#main_loop
#bck_btn
#second_btn
#second_btn2

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

#사람에 바운더리 사각형 그리기
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

#경고 띄우기
def draw_caution(frame, idxs):
    cv2.rectangle(frame, caut_cor_back, caut_cor_back2, (0, 0, 0), -1)
    cv2.circle(frame,(caut_cor[0]-45,caut_cor[1]-5),15,(0,0,255),-1)
    cv2.putText(frame, 'Caution : lack of workers'.format(len(idxs)), caut_cor, font, 
                caut_font_scale, caut_color, caut_thickness, cv2.LINE_AA)
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
            
            if classes_t[class_id] == "Safety Cone" and confidence > 0.8: # 신뢰도 임계값을 정한다.
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], 
                                                            frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
    
    # 중복되는 상자제거 필터링 NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

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

#작업인원수모니터링
def workers_counts_monitoring(num, term, auto_range):#num은 할당 인원수
    # 비디오 업로드
    cap = cv2.VideoCapture('videos/cone_people3.mp4')
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
                    if len(auto_boundary_tops)>0:
                        x0_m=auto_range_filter(auto_boundary_lefts,0,term)
                        y0_m=auto_range_filter(auto_boundary_tops,0,term)
                        w_m=auto_range_filter(auto_boundary_rights,1,term)-auto_range_filter(auto_boundary_lefts,0,term)
                        h_m=auto_range_filter(auto_boundary_bottoms,1,term)-auto_range_filter(auto_boundary_tops,0,term)
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
            frame=draw_caution(frame, idxs)

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
                frame = drawing(frame, idxs, boxes, term, pt1, pt2)
            elif len(auto_boundary_tops)>0:#보완이 필요한 부분
                frame = drawing(frame, idxs, boxes, term, pt1, pt2)    
            else:
                frame = drawing(frame, idxs, boxes, term, (0,0), (640,480))
        
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

#메인루프
def main_loop(btn,num1,num2,auto_range):
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

    if btn==2:
        print(auto_range)
        workers_counts_monitoring(num1,num2,auto_range)

def bck_btn():
    for widget in root.winfo_children():
        widget.destroy()
    button1 = tk.Button(root, text="접근금지구역 모니터링")
    button2 = tk.Button(root, text="작업인원수 모니터링",command=second_btn)
    button3 = tk.Button(root, text="적치물 제한 구역 모니터링")

    button1.place(x=50, y=70, width=250, height=50)
    button2.place(x=50, y=170, width=250, height=50)
    button3.place(x=50, y=270, width=250, height=50)

def second_btn():
    for widget in root.winfo_children():
        widget.destroy()
    button4 = tk.Button(root, text="뒤로 가기", command=bck_btn)
    button4.place(x=30, y=30, width=60, height=30)

    tk.Label(root,text='[작업인원수 모니터링]').place(x=50, y=80)
    tk.Label(root,text='1. 할당인원 수를 입력하면, 할당인원수에 미달 할 때 경고 합니다.').place(x=50, y=110)
    tk.Label(root,text='2. 할당인원 수는 2명 이상 10명 이하로 설정 할 수 있습니다.').place(x=50, y=140)
    
    tk.Label(root,text='할당인원 수 설정 : ').place(x=50, y=190)
    scale1 = tk.Scale(root, from_=2, to=10,orient="horizontal", length=200)
    scale1.place(x=160, y=170,)
    
    tk.Label(root,text='3. 사용자 기기의 성능에 따라 모니터링 빈도를 조절하세요.').place(x=50, y=240)
    tk.Label(root,text='4. 예를들어 10을 설정하면, 1초에 10번 모니터링 하게 됩니다.').place(x=50, y=270)
    tk.Label(root,text='5. 8보다 작은 경우, 사람의 경계를 표시하는 바운더리 박스가 그려지지 않습니다.').place(x=50, y=300)

    tk.Label(root,text='모니터링 빈도 : ').place(x=50, y=350)
    scale2 = tk.Scale(root, from_=1, to=60,orient="horizontal", length=300)
    scale2.set(10)
    scale2.place(x=160, y=330,)
    
    button5 = tk.Button(root, text="다 음",command= lambda : second_btn2(scale1.get(),scale2.get()))
    button5.place(x=450, y=200, width=150, height=50)

def second_btn2(allocated, term):
    for widget in root.winfo_children():
        widget.destroy()
    button4 = tk.Button(root, text="뒤로 가기", command=bck_btn)
    button4.place(x=30, y=30, width=60, height=30)

    tk.Label(root,text='[범위 자동 할당 설정]').place(x=50, y=80)
    tk.Label(root,text='1. 자동 할당 기능을 켜는 경우 라바콘을 자동으로 인식하여 범위를 자동으로 할당합니다.').place(x=50, y=110)
    tk.Label(root,text='2. 자동할당에 실패할 경우 수동범위조작모드로 변경됩니다.').place(x=50, y=140)
    tk.Label(root,text='3. 수동범위조작모드에서 마우스로 드래그하여 범위를 설정합니다.').place(x=50, y=170)
    tk.Label(root,text='4. 마우스로 드래그 할 때 파란색이 되어야 범위가 설정됩니다. 너무 작으면 안 됩니다.').place(x=50, y=200)
    tk.Label(root,text='5. 범위는 u 를 눌러서 초기화 할 수 있습니다.').place(x=50, y=230)
    tk.Label(root,text='6. r을 누르면 다시 자동으로 범위를 할당 합니다.').place(x=50, y=260)
    tk.Label(root,text='7. 할당된 이후 드래그 하여 다시 범위를 설정 할 수 있습니다.').place(x=50, y=290)
    tk.Label(root,text='범위 자동 할당[ON/OFF]').place(x=50, y=320)

    tk.Label(root, text="ON").place(x=200, y=310)
    toggle_var = tk.IntVar()
    toggle_check = tk.Checkbutton(root,variable=toggle_var)
    toggle_check.place(x=200, y=325)

    tk.Label(root,text='종료시에는 ESC키를 눌러서 종료해야 합니다.').place(x=50, y=360)

    button6 = tk.Button(root, text="모니터링 시작",command= lambda : main_loop(2,allocated,term,toggle_var.get()))
    button6.place(x=450, y=300, width=150, height=50)



#인터페이스 정의
root = tk.Tk()
root.title("Monitoring system")
root.geometry("640x400")
root.resizable(False, False)

button1 = tk.Button(root, text="접근금지구역 모니터링")
button2 = tk.Button(root, text="작업인원수 모니터링",command=second_btn)
button3 = tk.Button(root, text="적치물 제한 구역 모니터링")

button1.place(x=50, y=70, width=250, height=50)
button2.place(x=50, y=170, width=250, height=50)
button3.place(x=50, y=270, width=250, height=50)

root.mainloop()#인터페이스 실행


