import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from src.gesture_detector import GestureDetector
from src.gesture_classifier import GestureClassifier
from src.system_controller import SystemController
from src.cursor_controller import CursorController
from src.hand_tracker import HandTracker
from utils.fps_counter import FPSCounter
from utils.visualizer import Visualizer
from config.settings import Settings

def main(args):
    settings = Settings()
hand_tracker = HandTracker(max_hands=settings.MAX_HANDS,
detection_confidence=settings.DETECTION_CONFIDENCE,
tracking_confidence=settings.TRACKING_CONFIDENCE
)
gesture_detector = GestureDetector()
classifier = GestureClassifier(model_type=args.model)
sys_controller = SystemController()
cursor_ctrl = CursorController(smoothing_alpha=settings.CURSOR_SMOOTHING)
fps_counter = FPSCounter()
visualizer = Visualizer()
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
print("\n" + "="*60)
print(" Hand Gesture HCI System | Press Q to quit")
print("="*60)
print(f" Model : {args.model.upper()} | Camera: {args.camera}")
print(f" Cursor : Point (index finger alone) to move cursor")
print(f" Pinch to click | Hold pinch + move to drag")
print(f" Peace sign to scroll")
print("="*60 + "\n")
prev_gesture = None
gesture_stable = 0
while True:
    ret, frame = cap.read()
if not ret:
    print("[ERROR] Cannot read from camera.")
break
frame = cv2.flip(frame, 1)
fps = fps_counter.update()
results, landmarks_list = hand_tracker.process(frame)
cursor_active = False
if landmarks_list:
    for hand_landmarks, handedness in landmarks_list:
        raw_gesture = gesture_detector.detect(hand_landmarks, handedness)
features = gesture_detector.extract_features(hand_landmarks)
ml_gesture = classifier.predict(features)
final_gesture = ml_gesture if ml_gesture != "unknown" else raw_gesture

# Real-time cursor — runs every frame, no stability gate
cursor_ctrl.process_frame(hand_landmarks, final_gesture)
if cursor_ctrl.cursor_mode:
    cursor_active = True
# Discrete system actions — only when NOT in cursor mode
if not cursor_ctrl.cursor_mode:
    if final_gesture == prev_gesture:
        gesture_stable += 1
else:
    gesture_stable = 0
prev_gesture = final_gesture
if gesture_stable == settings.STABLE_FRAMES:
    result = sys_controller.execute(final_gesture, hand_landmarks)
if result not in ("cooldown", "no action"):
    print(f" Gesture: {final_gesture:<20} -> {result}")
else:
    gesture_stable = 0
prev_gesture = None
visualizer.draw_hand(
frame, hand_landmarks, handedness,
final_gesture, gesture_stable,
cursor_mode=cursor_ctrl.cursor_mode,
cursor_info=cursor_ctrl.get_info()
)
visualizer.draw_hud(
frame, fps, args.model,
sys_controller.get_status(),
cursor_active=cursor_active,
cursor_info=cursor_ctrl.get_info()
)
cv2.imshow("Hand Gesture HCI", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
#break
cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Session ended. Goodbye!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture HCI System")
parser.add_argument("--model", type=str, default="rf",
choices=["rf", "cnn"], help="Classifier: rf | cnn")
parser.add_argument("--camera", type=int, default=0,
help="Camera index (default: 0)")
args = parser.parse_args()

main(args)
