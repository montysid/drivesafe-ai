from enum import Enum
from config import HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD, DISTRACTION_FRAMES, CONSECUTIVE_FRAMES, EAR_THRESHOLD, YAWN_THRESHOLD, YAWN_FRAMES


class DriverState(Enum):
    ALERT = 0
    DROWSY = 1
    YAWNING = 2
    DISTRACTED = 3


class DrowsinessDetector:

    def __init__(self):
        self.drowsiness_frames = 0
        self.distraction_frames = 0
        self.yawning_frames = 0

    # NOTE: This rule-based alert logic is fragile. MediaPipe landmark noise, eye appearance changes
    # from mouth movement, and brief pose loss can cause the wrong alert to appear.
    # This is exactly why a machine learning approach is needed for more robust detection.
    def detect(self, ear, yaw, pitch, mouth_ar):
        """Detect driver state based on eye aspect ratio, head pose, and mouth opening"""

        # Check yawning first
        if mouth_ar > YAWN_THRESHOLD:
            self.yawning_frames += 1
            self.distraction_frames = 0
            self.drowsiness_frames = 0

            if self.yawning_frames >= YAWN_FRAMES:
                return DriverState.YAWNING

            return DriverState.ALERT

        self.yawning_frames = 0

        # Check distraction next (head turned away or tilted up/down)
        if abs(yaw) > HEAD_YAW_THRESHOLD or abs(pitch) > HEAD_PITCH_THRESHOLD:
            self.distraction_frames += 1
            self.drowsiness_frames = 0

            if self.distraction_frames >= DISTRACTION_FRAMES:
                return DriverState.DISTRACTED

            return DriverState.ALERT
        
        self.distraction_frames = 0

        # Check for drowsiness (eyes closed)
        if ear < EAR_THRESHOLD:
            self.drowsiness_frames += 1

            if self.drowsiness_frames >= CONSECUTIVE_FRAMES:
                return DriverState.DROWSY

            return DriverState.ALERT
        
        self.drowsiness_frames = 0

        return DriverState.ALERT