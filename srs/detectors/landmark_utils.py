import cv2
import numpy as np
from config import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_LANDMARKS, POSE_LANDMARKS


class LandmarkUtils:

    @staticmethod
    def calculate_distance(point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    @staticmethod
    def calculate_eye_aspect_ratio(eye_landmarks):
        """Calculate eye aspect ratio (EAR) from eye landmarks"""
        # Calculate vertical distances
        vertical1 = LandmarkUtils.calculate_distance(eye_landmarks[1], eye_landmarks[5])
        vertical2 = LandmarkUtils.calculate_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Calculate horizontal distance
        horizontal = LandmarkUtils.calculate_distance(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    @staticmethod
    def get_eye_aspect_ratio(landmarks, frame_width, frame_height):
        """Get the average EAR for both eyes"""
        # Get left eye landmarks
        left_eye = []
        for idx in LEFT_EYE_INDICES:
            lm = landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            left_eye.append([x, y])
        
        # Get right eye landmarks
        right_eye = []
        for idx in RIGHT_EYE_INDICES:
            lm = landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            right_eye.append([x, y])
        
        # Calculate EAR for both eyes
        left_ear = LandmarkUtils.calculate_eye_aspect_ratio(left_eye)
        right_ear = LandmarkUtils.calculate_eye_aspect_ratio(right_eye)
        
        # Average EAR
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear

    @staticmethod
    def get_mouth_aspect_ratio(landmarks, frame_width, frame_height):
        """Get the mouth aspect ratio for yawning detection"""
        mouth_points = []
        for idx in MOUTH_LANDMARKS:
            lm = landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            mouth_points.append([x, y])

        vertical = LandmarkUtils.calculate_distance(mouth_points[0], mouth_points[1])
        horizontal = LandmarkUtils.calculate_distance(mouth_points[2], mouth_points[3])

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    @staticmethod
    def get_head_pose(
        landmarks,
        frame_width,
        frame_height
    ):

        image_points = []

        for idx in POSE_LANDMARKS:

            lm = landmarks[idx]

            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)

            image_points.append([x, y])

        image_points = np.array(
            image_points,
            dtype=np.float64
        )

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        focal_length = frame_width

        camera_matrix = np.array([
            [focal_length, 0, frame_width / 2],
            [0, focal_length, frame_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1))
            )

            if not success:
                print("solvePnP failed")
                return 0.0, 0.0, 0.0

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                rotation_matrix
            )

            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]
            
            # Handle NaN values
            if np.isnan(pitch) or np.isnan(yaw) or np.isnan(roll):
                print("NaN values detected in angles")
                return 0.0, 0.0, 0.0
            
            # Clamp angles to reasonable ranges
            pitch = np.clip(pitch, -90, 90)
            yaw = np.clip(yaw, -90, 90)
            roll = np.clip(roll, -90, 90)

            return pitch, yaw, roll
        
        except Exception as e:
            print(f"Error in get_head_pose: {e}")
            return 0.0, 0.0, 0.0