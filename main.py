import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

#mediapipe hand track setup
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,min_tracking_confidence=0.5)

#webcam setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if success==False:
        break
    
    frame=cv2.flip(frame, 1)
    img_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process the frame
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #drawing landmarks using our direct import
            mp_drawing.draw_landmarks(frame,hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Level 1: Tracking', frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()