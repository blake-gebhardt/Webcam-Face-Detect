import cv2
import os

# Set the directory for storing images
dir_name = "dataset"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Initialize the camera
cam = cv2.VideoCapture(0)

# Set the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Set the size of the images
img_width, img_height = 150, 150

# Set the counter for the number of images
count = 0

# Start capturing images
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces and save the images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        count += 1
        file_name = os.path.join(dir_name, f"image_{count}.jpg")
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (img_width, img_height))
        cv2.imwrite(file_name, roi_gray)

    # Display the captured images
    cv2.imshow("Face Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
