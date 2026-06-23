import cv2

from src.video.video_stream import VideoStream
from src.detectors.face_detector import FaceDetector
from src.detectors.landmark_utils import LandmarkUtils
from src.detectors.drowsiness_detector import DrowsinessDetector, DriverState
from src.alerts.alert_manager import AlertManager


class DriveSafeApp:

    def __init__(self):

        self.video = VideoStream()

        self.face_detector = FaceDetector()

        self.drowsiness_detector = DrowsinessDetector()

        self.alert_manager = AlertManager()
        
        self.last_yaw = 0.0  # Track last known yaw for smoothing
        self.last_pitch = 0.0  # Track last known pitch for smoothing
        self.last_ear = 1.0  # Track last known EAR (assume eyes open)
        self.last_mouth_ar = 0.0  # Track last known mouth aspect ratio

    def run(self):

        while True:

            success, frame = self.video.read()

            if not success:
                continue

            frame_height, frame_width, _ = frame.shape

            results = self.face_detector.detect(frame)

            state = None

            if results.multi_face_landmarks:

                try:
                    landmarks = (
                        results.multi_face_landmarks[0]
                        .landmark
                    )

                    # Get head pose
                    pitch, yaw, roll = (
                        LandmarkUtils.get_head_pose(
                            landmarks,
                            frame_width,
                            frame_height
                        )
                    )
                    
                    # Get eye aspect ratio
                    ear = LandmarkUtils.get_eye_aspect_ratio(
                        landmarks,
                        frame_width,
                        frame_height
                    )
                    
                    # Get mouth aspect ratio for yawning detection
                    mouth_ar = LandmarkUtils.get_mouth_aspect_ratio(
                        landmarks,
                        frame_width,
                        frame_height
                    )
                    
                    # Update last known pose values
                    self.last_pitch = pitch
                    self.last_yaw = yaw
                    self.last_ear = ear
                    self.last_mouth_ar = mouth_ar
                    
                    # Detect driver state
                    state = self.drowsiness_detector.detect(
                        ear,
                        yaw,
                        pitch,
                        mouth_ar
                    )

                    # Display metrics for debugging
                    cv2.putText(
                        frame,
                        f"EAR: {ear:.2f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:.2f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:.2f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    cv2.putText(
                        frame,
                        f"MAR: {mouth_ar:.2f}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    
                    # Display face detection status
                    cv2.putText(
                        frame,
                        "Face: DETECTED",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        1
                    )

                    # Draw alert for current state
                    self.alert_manager.draw(
                        frame,
                        state
                    )
                    
                except Exception as e:
                    print(f"Error processing landmarks: {e}")
                    cv2.putText(
                        frame,
                        "Face: ERROR",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        1
                    )
            else:
                # Face lost - use last known pose for distraction detection continuity.
                # Failure mode: if the face disappears during a slight turn, the system can miss the distraction.
                print("Face lost - using last known pose for tracking")
                state = self.drowsiness_detector.detect(
                    self.last_ear,
                    self.last_yaw,
                    self.last_pitch,
                    self.last_mouth_ar
                )
                
                # Draw alert if state is still detected
                self.alert_manager.draw(
                    frame,
                    state
                )
                
                cv2.putText(
                    frame,
                    "Face: NOT DETECTED",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )

            cv2.imshow(
                "DriveSafe AI",
                frame
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        self.video.release()

        cv2.destroyAllWindows()