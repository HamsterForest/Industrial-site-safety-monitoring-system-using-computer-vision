import cv2


cap = cv2.VideoCapture(0)

while True:
    ret,img_color = cap.read()

    if ret == False:#캠이 없으면 종료
        continue

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)#GrayScale영상

    cv2.imshow("Color", img_color)
    cv2.imshow("Gray", img_gray)

    if cv2.waitKey(1)&0xFF == 27:#esc키 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()