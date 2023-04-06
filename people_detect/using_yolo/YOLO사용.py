import cv2
import numpy as np
import time

# yolo 로드
net = cv2.dnn.readNet("weight_files_folder/yolov3.weights", "people_detect/using_yolo/yolov3.cfg")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers() if isinstance(i, list)]

# 비디오 업로드
cap = cv2.VideoCapture('people_detect/vtest.avi')

# Define classes
classes = []
with open("people_detect/using_yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize counters

people_count = 0

# Set up text font and position
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
color = (255, 0, 0)
thickness = 2

#fps 조절
prev_time=0
FPS=10
initial_flag=True

while True:
    #fps 조절
    if initial_flag==True:
        ret, frame = cap.read()
        initial_flag=False

    current_time = time.time() - prev_time
    if current_time > 1./ FPS:
        prev_time = time.time()
        ret, frame = cap.read()

    if not ret:
        break

    people_count = 0
    
    # 인식할 수 있는 blob으로 변환
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    #노이즈 제거를 위해 선언함 boxes에는 상자의 위치 좌표가 표시되고 confidences는 각 boxes에 대한 confidence
    confidences=[]
    boxes=[]

    # Process each output layer
    for output in outputs:
        # Process each detection
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Check if object is a person
            if classes[class_id] == "person" and confidence > 0.5:
                left, top, w, h = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                left, top, w, h = int(left - w/2), int(top - h/2), int(w), int(h)
                confidences.append(float(confidence))
                boxes.append([left,top,w,h])
                
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs)>0:
        for i in idxs.flatten():
            people_count+=1
            box=boxes[i]
            left=box[0]
            top=box[1]
            w=box[2]
            h=box[3]
            # Draw bounding box around person
            cv2.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)

    # Add people count text on frame
    cv2.putText(frame, 'People Count: {}'.format(people_count), org, font, font_scale, color, thickness, cv2.LINE_AA)

    # Display frame with bounding boxes and people count
    cv2.imshow('Frame', frame)
    
    # Exit if "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
