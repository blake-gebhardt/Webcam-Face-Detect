import cv2
import numpy as np
import os

# Set the directories for the dataset and the trained model
dataset_dir = "dataset"
model_file = "blake_model.yml"

# Set the size of the images
img_width, img_height = 150, 150

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Create an empty list for the training data and labels
training_data = []
labels = []

# Load the images from the dataset and label them as "Blake"
for file_name in os.listdir(dataset_dir):
    if not file_name.endswith(".jpg"):
        continue
    img_path = os.path.join(dataset_dir, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    training_data.append(img)
    labels.append(0)

# Convert the training data and labels to numpy arrays
training_data = np.asarray(training_data)
labels = np.asarray(labels)

# Train the face recognition model using LBPH algorithm
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(training_data, labels)

# Save the trained model to a file
face_recognizer.write(model_file)

# Initialize the camera
cam = cv2.VideoCapture(0)

# Start recognizing faces
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Identify the faces as "Blake" or "Not Blake" using the trained model
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (img_width, img_height))
        label, confidence = face_recognizer.predict(roi_gray)
        if label == 0:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Blake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Not Blake", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the recognized faces
    cv2.imshow("Face Recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
