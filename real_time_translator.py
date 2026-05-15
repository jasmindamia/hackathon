import cv2
import pickle
import numpy as np
import collections
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

#load model
try:
    file=open('./model.p', 'rb')
    model_dict=pickle.load(file)
    file.close()
    model=model_dict['model']
except:
    print("Error: 'model.p' not found. Train your model first!")
    exit()

#setapp
hands=mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
history_list=[]

print("Level 4: Word Translator Active. (Ghost Guard Enabled)")

while True:
    success,frame=cap.read()
    if success==False:
        break
    frame=cv2.flip(frame, 1)
    img_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=hands.process(img_rgb)

    current_coords=[]
    for zero in range(84):
        current_coords.append(0.0)
    
    #keluar bila detect tangan je, kalau tak detect tangan, clear memory so bila tangan masuk balik, dia tak predict based on old posisiont
    if results.multi_hand_landmarks != None:
        hand_index=0
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            lbl=results.multi_handedness[hand_index].classification[0].label
            if lbl=="Right":
                start_idx=0
            else: 
                start_idx=42
            coords=[]
            for lm in hand_lms.landmark:
                coords.append(lm.x)
                coords.append(lm.y)
            for position in range(42):
                current_coords[start_idx+position]=coords[position]
            
            hand_index=hand_index+1

        #add to memorty
        history_list.append(current_coords)
        
        # FIX: Keep history size at exactly 11 items maximum
        if len(history_list) > 11:
            history_list.pop(0)

        #predict only if memory is full AND hands are still visible
        if len(history_list)==11:
            motion_window=history_list[0]+history_list[5]+history_list[10]
            prediction=model.predict([np.asarray(motion_window)])
            predicted_word=prediction[0]

            #display word
            cv2.putText(frame, str(predicted_word),(50,80), 
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    else:
        history_list.clear()

    cv2.imshow('Level 4: Word Translator', frame)
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
