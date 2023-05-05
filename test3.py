import cv2
 
isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)
 
def onMouse(event, x, y, flags, param):
    print('hello')
    global isDragging, x0, y0, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
        w = 0
        h = 0
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            w = x - x0
            h = y - y0
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
    

cap = cv2.VideoCapture('videos/vtest.avi')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', onMouse)

    if w>0 and h>0:
        cv2.rectangle(frame, (x0,y0), (x0+w,y0+h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)

    keycode=cv2.waitKey(25)
    if keycode == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
        
