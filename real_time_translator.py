import cv2
import pickle
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# 1. Load the trained "Brain" (The Pickle file)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# 2. Initialize MediaPipe
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("Real-time Translator Active! Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape # Get frame dimensions for text placement
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- THE PREDICTION LOGIC ---
            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)
            
            # Use the model to predict the word
            # We wrap data_aux in [] because the model expects a 2D array
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # 3. Display the result on the screen
            # cv2.putText(image, text, coordinates, font, scale, color, thickness)
            cv2.putText(frame, predicted_character, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Level 4: Real-Time Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()