import cv2
import csv
import collections
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- CONFIG ---
CURRENT_WORD = "TERIMA KASIH" 
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Buffer stores the last 11 frames of data
history = collections.deque(maxlen=11)

print(f"COLLECTING MOTION: {CURRENT_WORD}")
print("Press 's' to save a motion sequence. 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Reset current coordinates for this frame
    current_coords = [0.0] * 84

    if results.multi_hand_landmarks:
        # --- RESTORED: DRAWING SKELETONS ---
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Sort data into Right (0-41) or Left (42-83)
            lbl = results.multi_handedness[i].classification[0].label
            idx = 0 if lbl == "Right" else 42
            coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
            current_coords[idx:idx+42] = coords
    
    # Add current frame to history
    history.append(current_coords)

    cv2.imshow('Motion Collector', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(history) == 11:
        # Stack 3 frames (Start, Mid, End of the 0.3s window)
        motion_data = history[0] + history[5] + history[10]
        with open("hand_data.csv", "a", newline="") as f:
            csv.writer(f).writerow([CURRENT_WORD] + motion_data)
        print(f"Captured motion for: {CURRENT_WORD}")

    elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()