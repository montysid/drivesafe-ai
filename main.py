from __future__ import annotations

import cv2
import numpy as np
import winsound
import os
import threading
import time

print("Starting program...")

# Load cascade classifiers for face and eyes detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

print("Cascade classifiers loaded")

# Find available camera
print("Searching for available cameras...")
camera_index = None
for i in range(5):  # Try indices 0-4
    cap_test = cv2.VideoCapture(i)
    if cap_test.isOpened():
        print(f"Camera found at index {i}")
        camera_index = i
        cap_test.release()
        break
    cap_test.release()

if camera_index is None:
    print("Error: No camera found. Available indices tried: 0-4")
    print("Note: Your original code used index 1. You may need to connect a camera or check permissions.")
    exit()

print("Opening webcam...")
cap = cv2.VideoCapture(camera_index)

# Configure camera properties to improve stability
try:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
except:
    pass

if not cap.isOpened():
    print(f"Error: Could not open webcam at index {camera_index}")
    exit()

print(f"Webcam opened successfully at index {camera_index}")

# Load alarm sound file
alarm_file = "alarm.wav.mp3"

# Try to find the alarm file - check both relative and absolute paths
if not os.path.exists(alarm_file):
    abs_path = os.path.abspath(alarm_file)
    if os.path.exists(abs_path):
        alarm_file = abs_path
    else:
        print(f"Warning: Alarm file not found at '{alarm_file}' or '{abs_path}'")
        alarm_file = None
else:
    alarm_file = os.path.abspath(alarm_file)

alarm_playing = False
alarm_thread = None

def play_alarm_loop():
    """Play system beep alarm in a loop until stopped"""
    global alarm_playing
    try:
        while alarm_playing:
            # Play 1000 Hz beep for 300ms
            winsound.Beep(1000, 300)
            time.sleep(0.1)  # 100ms gap between beeps
    except Exception as e:
        print(f"Error in alarm thread: {e}")

if os.path.exists(alarm_file):
    print(f"Alarm sound loaded: {alarm_file}")
else:
    print(f"Warning: Alarm file not found: {alarm_file}")
    alarm_file = None

# Define the Eye Aspect Ratio (EAR) threshold for drowsiness
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 10  # Number of consecutive frames for drowsiness detection

frame_count = 0
frame_skip_count = 0
max_frame_skips = 5  # Try only 5 times before giving up

while True:
    success, frame = cap.read()

    if not success:
        frame_skip_count += 1
        
        if frame_skip_count > max_frame_skips:
            print("Camera disconnected. Attempting to reconnect...")
            cap.release()
            cv2.destroyAllWindows()
            
            # Try to reconnect
            time.sleep(1)
            cap = cv2.VideoCapture(camera_index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print("Failed to reconnect to camera")
                break
            
            print("Reconnected to camera")
            frame_skip_count = 0
        
        cv2.waitKey(100)
        continue
    
    frame_skip_count = 0  # Reset counter on successful frame

    # Convert frame to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Region of interest (ROI) for the face
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detect eyes in the face
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) < 2:
            # Eyes not detected - possible drowsiness
            frame_count += 1
            
            # Draw WHITE box around head
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
            
            if frame_count > CONSECUTIVE_FRAMES:
                # Start alarm if not already playing
                if not alarm_playing and alarm_file:
                    alarm_playing = True
                    alarm_thread = threading.Thread(target=play_alarm_loop, daemon=True)
                    alarm_thread.start()
                    print("ALARM ACTIVATED - Eyes closed detected!")
                
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3,
                )
        else:
            # Eyes detected - reset counter
            frame_count = 0
            
            # Stop alarm when eyes are detected
            if alarm_playing:
                alarm_playing = False
                if alarm_thread:
                    alarm_thread.join(timeout=0.5)
                print("Alarm stopped - Eyes detected again")
            
            # Draw normal blue rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw rectangles around eyes
        for i, (ex, ey, ew, eh) in enumerate(eyes):
            if i < 2:  # Only show first 2 eyes
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                )

    # Display frame counter and eye detection status
    cv2.putText(
        frame,
        f"Frame Count: {frame_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Press 'q' or ESC to quit
    key = cv2.waitKey(1)
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Stop alarm if still playing
if alarm_playing:
    alarm_playing = False
    if alarm_thread:
        alarm_thread.join(timeout=0.5)

print("Program ended")