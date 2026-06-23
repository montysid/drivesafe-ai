# NOTES

## Test observations and failure cases

- When I close my eyes, the display often stays white and does not reliably switch to red `DROWSY`. That means drowsiness detection is not triggering even though the eyes are closed long enough.
- When I look away for 2 seconds, the yellow `DISTRACTION DETECTED` alert often does not appear. The system is losing the face mesh or not maintaining a stable head pose during turning.
- A fake yawn (opening my mouth) sometimes causes the system to show `DROWSINESS DETECTED` instead of `YAWNING DETECTED`. This indicates the current mouth / eye rules are fragile and mouth opening can change eye landmark values.
- The system is very sensitive to head angle. Slight turns or looking to the side can make the face detection disappear, which prevents distraction detection.
- Lighting changes, glasses, or reflections make the face mesh less reliable. Under bad lighting the landmarks jump or disappear.
- The current detection is purely rule-based and struggles with noisy MediaPipe landmarks, especially for eyes and mouth.

