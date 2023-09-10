import cv2
import numpy as np
from scipy.signal import find_peaks

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

heart_rate_values = []

roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read a frame from the camera.")
        break

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    face_net.setInput(blob)
    detections = face_net.forward()

    face_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            face_detected = True
            break

    if face_detected:
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        average_intensity = np.mean(gray_roi)
        heart_rate_values.append(average_intensity)

    if len(heart_rate_values) > 10 and face_detected:
        heart_rate_signal = np.array(heart_rate_values)
        heart_rate_signal = (heart_rate_signal - np.mean(heart_rate_signal)) / np.std(heart_rate_signal)
        peaks, _ = find_peaks(heart_rate_signal, height=0.5)
        
        heart_rate_bpm = len(peaks) / (len(heart_rate_signal) / cap.get(cv2.CAP_PROP_FPS)) * 60
        cv2.putText(frame, f"Heart Rate: {heart_rate_bpm:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
