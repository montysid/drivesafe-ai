import cv2
import mediapipe as mp

class FaceDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        # Use refined landmarks and lower min detection confidence for better tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        self.last_yaw = 0.0  # Store last yaw for smoothing

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        return results
    
    def get_last_yaw(self):
        """Get the last detected yaw angle for smoothing"""
        return self.last_yaw
    
    def set_last_yaw(self, yaw):
        """Store the current yaw for future smoothing"""
        self.last_yaw = yaw