import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.ops.signal.shape_ops import frame

model = load_model('LSTM_CNN.h5')
emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +  "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret,  frame = cap.read()
    if not ret:
        break

    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50,50))


    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (28, 28))
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_resized = face_resized / 255.0

        predictions = model.predict(face_resized)
        label_index = np.argmax(predictions)
        emotion = emotions[label_index]


        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,0), 2)
        cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow('Facial Expression ', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()