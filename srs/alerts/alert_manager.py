import cv2


class AlertManager:

    def __init__(self):
        self.drowsy_count = 0
        self.distraction_count = 0
        self.last_state = None

    def draw(self, frame, state):
        """Draw alert on frame based on driver state and maintain counts."""

        if state is None:
            return

        if state == self.last_state:
            new_event = False
        else:
            new_event = True

        if new_event:
            if state.name == "DROWSY":
                self.drowsy_count += 1
            elif state.name == "DISTRACTED":
                self.distraction_count += 1

        self.last_state = state

        # Always show event counts
        cv2.rectangle(frame, (10, 10), (360, 65), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Drowsy: {self.drowsy_count}  Distracted: {self.distraction_count}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if state.name == "DISTRACTED":
            cv2.rectangle(frame, (20, 80), (560, 140), (0, 255, 255), -1)
            cv2.putText(
                frame,
                "DISTRACTION DETECTED",
                (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2
            )

        elif state.name == "DROWSY":
            cv2.rectangle(frame, (20, 80), (520, 140), (0, 0, 255), -1)
            cv2.putText(
                frame,
                "DROWSINESS DETECTED",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2
            )

        elif state.name == "YAWNING":
            cv2.rectangle(frame, (20, 80), (520, 140), (0, 165, 255), -1)
            cv2.putText(
                frame,
                "YAWNING DETECTED",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2
            )

        elif state.name == "ALERT":
            frame_height, frame_width = frame.shape[:2]
            cv2.rectangle(frame, (10, frame_height - 50), (180, frame_height - 10), (0, 255, 0), -1)
            cv2.putText(
                frame,
                "ALERT",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
