import cv2

# Load the Haar Cascade Classifier for detecting people
face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Create a background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Open a video capture object and start reading frames
cap = cv2.VideoCapture(0)

# Define a variable to keep track of the number of people
num_people = 0

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction to the frame
    fg_mask = bg_subtractor.apply(frame)

    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Detect people in the foreground mask
    faces = face_cascade.detectMultiScale(fg_mask)

    # Draw bounding boxes around the detected people
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Update the number of people in the frame
    num_people = len(faces)

    # Display the number of people in the frame
    cv2.putText(frame, f'Number of people: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
