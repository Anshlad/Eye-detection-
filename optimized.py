import cv2
import time
import winsound
import threading

# Load the cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

eye_closed_start_time = None
eye_closed_duration_threshold = 3  # seconds

def play_sound():
    winsound.Beep(1000, 1000)  # Frequency: 1000 Hz, Duration: 1000 ms (1 second)

def process_frame(frame):
    global eye_closed_start_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) == 0:
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()
        else:
            eye_closed_duration = time.time() - eye_closed_start_time
            if eye_closed_duration > eye_closed_duration_threshold:
                play_sound()
                eye_closed_start_time = time.time()  # Reset the timer after playing the sound
    else:
        eye_closed_start_time = None

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    return frame

def capture_and_process():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        cv2.imshow('Eye Detector', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_thread = threading.Thread(target=capture_and_process)
capture_thread.start()
capture_thread.join()
