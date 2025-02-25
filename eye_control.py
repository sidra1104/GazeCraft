import time
import winsound
import cv2
import mediapipe as mp
import numpy as np

def beep_every_second():
    while True:
        winsound.Beep(1000, 500)  # 1000 Hz frequency, 500 ms duration
        time.sleep(1)

def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_eye_closure():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])
                    h, w, _ = frame.shape
                    landmarks[:, 0] *= w
                    landmarks[:, 1] *= h

                    left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
                    right_eye = landmarks[[362, 385, 387, 263, 373, 380]]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < 0.2:
                        cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Eye Closure Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()