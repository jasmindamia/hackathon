import cv2
import mediapipe as mp
import csv
import os

# --- INITIALIZATION ---
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# The name of your "Notebook" file
DATA_FILE = "hand_data.csv"

# --- THE COLLECTION SETTINGS ---
# Change this label every time you want to record a new sign!
current_label = "ABSOLUTE CINEMA" 

print(f"Ready to record for: {current_label}")
print("Press 's' to SAVE a frame. Press 'q' to QUIT.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- THE DATA SAVING LOGIC ---
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                # 1. Create a list to hold the data for this frame
                data_row = []
                data_row.append(current_label) # First column is the label
                
                # 2. Extract x and y for all 21 points
                for lm in hand_landmarks.landmark:
                    data_row.append(lm.x)
                    data_row.append(lm.y)
                
                # 3. Write to the CSV file
                with open(DATA_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data_row)
                
                print(f"Saved 1 frame of '{current_label}'")

    cv2.imshow('Level 2: Data Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()