import cv2
import csv
import collections
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

#congifigurate
CURRENT_WORD = "test" 
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

#0.3 seconds, 11 frames per pressing s for one time
history_list=[]

print(f"COLLECTING MOTION: {CURRENT_WORD}")
print("Press 's' to save a motion sequence. 'q' to quit.")

while cap.isOpened():
    success, frame=cap.read()
    if success==False:
        break

    frame=cv2.flip(frame,1)

    img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(img_rgb)

    #coordinate reset
    current_coords=[]
    for i in range(84): current_coords.append(0.0)

    if results.multi_hand_landmarks:
        #drawing lines on hands n getting coords
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame,hand_lms,mp_hands.HAND_CONNECTIONS)
            
            #sort data into right (0-41) or left (42-83)
            #satu tangan ada 21 joints, each coordinte x n y, so 2 tangan is 84
            lbl = results.multi_handedness[i].classification[0].label
            if lbl=="Right":
                start_idx=0
            else: 
                start_idx=42
            
            hand_coords=[]
            for lm in hand_lms.landmark:
                hand_coords.append(lm.x)
                hand_coords.append(lm.y)

            for j in range(42):
                current_coords[start_idx+j]=hand_coords[j]
    
    #add current frame to history
    history_list.append(current_coords)

    cv2.imshow('Motion Collector',frame)
    
    
    key=cv2.waitKey(1)

    if key==ord('q'):
        break

    if key== ord('s'):
        
        if len(history_list) >= 11:
            last_11_frames = history_list[-11:]
            motion_data=last_11_frames[0]+last_11_frames[5]+last_11_frames[10]

            with open("hand_data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                row_data=[CURRENT_WORD]+motion_data
                writer.writerow(row_data)
            
            print("Captured motion for: "+str(CURRENT_WORD))

    elif key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
