import cv2
import pickle
import numpy as np
import collections
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 1. Load the Model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: 'model.p' not found. Train your model first!")
    exit()

# 2. Setup
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
history = collections.deque(maxlen=11)

print("Level 4: Word Translator Active. (Ghost Guard Enabled)")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    current_coords = [0.0] * 84
    
    # --- GHOST GUARD: ONLY PROCESS IF HANDS ARE SEEN ---
    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            lbl = results.multi_handedness[i].classification[0].label
            idx = 0 if lbl == "Right" else 42
            coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
            current_coords[idx:idx+42] = coords
        
        # Add to memory
        history.append(current_coords)

        # 3. Predict only if memory is full AND hands are still visible
        if len(history) == 11:
            motion_window = history[0] + history[5] + history[10]
            prediction = model.predict([np.asarray(motion_window)])
            predicted_word = prediction[0]

            # Display the word
            cv2.putText(frame, str(predicted_word), (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        # If no hands are seen, clear the memory buffer so it doesn't 
        # use "old" hand positions when you bring your hand back.
        history.clear()

    cv2.imshow('Level 4: Word Translator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()