import cv2

# Create Haar Cascade classifier for detecting people
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Create video capture object
cap = cv2.VideoCapture(1)

while True:
    # Read frame from video capture object
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people in the frame using the Haar Cascade classifier
    people = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Update people count only if there are people in the frame
    if len(people) > 0:
        people_count = len(people)
    else:
        people_count = 0

    # Loop over the people and draw rectangles around them
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame with people count
    cv2.putText(frame, "People Count: {}".format(people_count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Frame", frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()
