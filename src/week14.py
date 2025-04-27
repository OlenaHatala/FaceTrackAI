import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import joblib

cnn_model = load_model('weights/cnn_face_classifier_EPOCHS_5.keras')  
rf_model = joblib.load('weights/rf_model_new.joblib')
scaler = joblib.load('weights/scaler_new.joblib')
yolo_model = YOLO('weights/yolov8n_face_best.pt')

cap = cv2.VideoCapture('data/video/security_camera.mp4')  

if not cap.isOpened():
    print("❗ Error: Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  

def detect_faces_yolo(frame):
    results = yolo_model(frame)[0]
    faces = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box[:4]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        faces.append((x, y, w, h))
    return faces

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized_cnn = cv2.resize(frame, (64, 64))
    frame_normalized_cnn = frame_resized_cnn / 255.0
    frame_input_cnn = np.expand_dims(frame_normalized_cnn, axis=0)  # (1, 64, 64, 3)

    cnn_prediction = cnn_model.predict(frame_input_cnn, verbose=0)
    cnn_label = "Face" if np.argmax(cnn_prediction) == 1 else "No Face"

    frame_resized_rf = cv2.resize(frame, (64, 64))
    frame_flat_rf = frame_resized_rf.flatten().reshape(1, -1)
    frame_flat_rf_scaled = scaler.transform(frame_flat_rf)

    rf_prediction = rf_model.predict(frame_flat_rf_scaled)
    rf_label = "Face" if rf_prediction[0] == 1 else "No Face"

    faces_yolo = detect_faces_yolo(frame)

    for (x, y, w, h) in faces_yolo:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  

    text = f'CNN: {cnn_label} | RF: {rf_label} | YOLO Faces: {len(faces_yolo)}'
    cv2.rectangle(frame, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    cv2.putText(frame, text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Simulated Real-Time Face Monitoring', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()