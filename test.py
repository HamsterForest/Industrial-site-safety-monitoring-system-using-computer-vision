import cv2
import numpy as np
import time

# yolo 로드
net = cv2.dnn.readNet("weight_files_folder/trafficcone/yolov3.weights", "weight_files_folder/trafficcone/yolov3.cfg")
#.weights => 훈련된 모델 파일, .cfg => 알고리즘 구성 파일

#output layer선언- 모든 레이어를 불러온 후 unconnected layer즉, output layer만 추린다.
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

img_color=cv2.imread('videos/trafficcone8.jpg',cv2.IMREAD_COLOR)

classes = []#감지 할 수 있는 모든 객체 명이 들어간다.
with open("weight_files_folder/trafficcone/obj.names", "r") as f:#.namses => 알고리즘이 감지 할 수 있는 객체의 이름 모음
    classes = [line.strip() for line in f.readlines()]


#영상에 글자를 넣기 위한 사전 설정
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
color = (255, 0, 0)
thickness = 2


# 이미지를 그대로 넣는 것이 아니라, blob으로 넣게 된다.
# blob은 이미지의 픽셀정보와 크기정보, 색의 채널 정보들을 가지는 행렬의 형태이다.
# blop의 사이즈가 클수록 accuracy가 높아지지만 연산 시간이 늘어나게 된다.
blob = cv2.dnn.blobFromImage(img_color, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)
print(outputs)
#노이즈 제거를 위해 선언함 boxes에는 상자의 위치 좌표가 표시되고 confidences는 각 boxes에 대한 confidence
confidences=[]
boxes=[]

for output in outputs:

    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        
        left, top, w, h = detection[:4] * np.array([img_color.shape[1], img_color.shape[0], img_color.shape[1], img_color.shape[0]])
        left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)

        confidences.append(float(confidence))
        boxes.append([left,top,w,h])
#중복되는 상자 제거
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(idxs)
if len(idxs)>0:
    for i in idxs.flatten():
        box=boxes[i]
        left=box[0]
        top=box[1]
        w=box[2]
        h=box[3]
        cv2.rectangle(img_color, (left, top), (left+w, top+h), (0, 255, 0), 2)


cv2.imshow('Frame', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

