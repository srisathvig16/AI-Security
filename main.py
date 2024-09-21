import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained mask detection model
model = load_model("/Users/srisathvig/Documents/Projects/5 - AI SECURITY/mask_detector_model.h5")  # Replace with your model's path

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function to preprocess the input face for the model
def preprocess_face(face):
    face = cv2.resize(face, (224, 224))  # Resize face to match model input size
    face = face.astype("float") / 255.0  # Normalize pixel values
    face = img_to_array(face)  # Convert to array
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()  # Read frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale (for face detection)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face from frame
        processed_face = preprocess_face(face)  # Preprocess the face
        
        # Predict if the person is wearing a mask or not
        (mask, without_mask) = model.predict(processed_face)[0]
        
        label = "Threat (Mask)" if mask > without_mask else "Safe (No Mask)"
        color = (0, 0, 255) if label == "Threat (Mask)" else (0, 255, 0)  # Red for threat, green for safe
        
        # Display the label and bounding box on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Show the frame with the results
    cv2.imshow('Mask Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()