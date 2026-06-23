from enum import Enum

from config import (
    EAR_THRESHOLD,
    CONSECUTIVE_FRAMES,
    HEAD_YAW_THRESHOLD,
    HEAD_PITCH_THRESHOLD,
    DISTRACTION_FRAMES,
    YAWN_THRESHOLD,
    YAWN_FRAMES
)


class DriverState(Enum):
    ALERT = "ALERT"
    DROWSY = "DROWSY"
    YAWNING = "YAWNING"
    DISTRACTED = "DISTRACTED"


class DrowsinessDetector:

    def __init__(self):

        self.drowsiness_frames = 0
        self.distraction_frames = 0
        self.yawning_frames = 0

    def detect(self, ear, yaw, pitch, mouth_ar):

        """
        Detects driver state based on EAR, head pose, and mouth aspect ratio.

        All counters are updated independently — no early returns, no cross-resets.

        Priority order: DROWSY > DISTRACTED > YAWNING > ALERT
        """

        # ── Step 1: Update ALL counters independently ──────────────────────────

        # Do NOT reset other counters here. A driver can be drowsy AND yawning.

        if ear < EAR_THRESHOLD:

            self.drowsiness_frames += 1

        else:

            self.drowsiness_frames = 0

        if abs(yaw) > HEAD_YAW_THRESHOLD or abs(pitch) > HEAD_PITCH_THRESHOLD:

            self.distraction_frames += 1

        else:

            self.distraction_frames = 0

        if mouth_ar > YAWN_THRESHOLD:

            self.yawning_frames += 1

        else:

            self.yawning_frames = 0

        # ── Step 2: Return most severe state (priority order matters) ──────────

        if self.drowsiness_frames >= CONSECUTIVE_FRAMES:

            return DriverState.DROWSY

        if self.distraction_frames >= DISTRACTION_FRAMES:

            return DriverState.DISTRACTED

        if self.yawning_frames >= YAWN_FRAMES:

            return DriverState.YAWNING

        return DriverState.ALERT