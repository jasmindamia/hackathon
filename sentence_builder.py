import cv2
import pickle
import numpy as np
import collections
import time
import os
from dotenv import load_dotenv
from groq import Groq
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

#load my keys
load_dotenv()
my_key=os.getenv("GROQ_API_KEY")
client=Groq(api_key=my_key)

#promps
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an MSL translator. Convert raw keywords into a formal, natural Malay sentence. Return ONLY the sentence. Example: 'SAYA SEKOLAH' -> 'Saya mahu pergi ke sekolah.'. Make it not more than 60 characters"
}

try:
    file=open('./model.p', 'rb')
    model_dict=pickle.load(file)
    file.close()
    model=model_dict['model']
except:
    print("Error: model.p not found!")
    exit()

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# bukak cam
cap = cv2.VideoCapture(0)

history_list=[]
sentence_list=[]
last_word=""
counter=0
final_output = "Llama System Ready..."
last_hand_time = time.time()
is_thinking = False

def get_llama_sentence(words):
    """Uses Llama 3 to refine the grammar."""
    if not words: return " "
    raw_input = " ".join(words)
    try:
        
        chat_completion=client.chat.completions.create(
            messages=[
                SYSTEM_PROMPT,
                {"role": "user", "content": f"Keywords: {raw_input}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.5
        )
        answer=chat_completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        print("Llama API Error:", e)
        return raw_input.capitalize()

while True:
    success,frame=cap.read()
    if success==False:
        break

    frame=cv2.flip(frame,1)

    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    current_coords=[]
    for zero in range(84):
        current_coords.append(0.0)
    
    if results.multi_hand_landmarks != None:
        last_hand_time = time.time()
        is_thinking = False

        for i in range(len(results.multi_hand_landmarks)):
            hand_lms=results.multi_hand_landmarks[i]
            mp_drawing.draw_landmarks(frame,hand_lms,mp_hands.HAND_CONNECTIONS)

            lbl= results.multi_handedness[i].classification[0].label

            if lbl=="Right":
                idx=0
            else:
                idx=42
            
            coords=[]
            for lm in hand_lms.landmark:
                coords.append(lm.x)
                coords.append(lm.y)
            
            for position in range(42):
                current_coords[idx+position]=coords[position]
        
        history_list.append(current_coords)
        if len(history_list)>11:
            history_list.pop(0)
        if len(history_list)==11:
            motion_window=history_list[0]+history_list[5]+history_list[10]
            
            array_data=np.asarray(motion_window)
            word=model.predict([array_data])[0]

            if word==last_word:
                counter=counter+1
            else:
                counter=0
                last_word=word

            if counter>=18:
                if len(sentence_list)==0 or word != sentence_list[-1]:
                    sentence_list.append(word)
                counter=0

                all_words=""
                for w in sentence_list:
                    all_words=all_words +w+" "
                final_output="input: " + all_words.strip()
    else:
        silence_gap=time.time() - last_hand_time
        if silence_gap>2.5 and len(sentence_list)>0 and is_thinking==False:
            final_output = "Llama is thinking..."
            
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame,(0,420),(640,480),(0,0,0),-1)
            cv2.putText(temp_frame,final_output,(20,460),1,1.2,(0,255,255),2)
            cv2.imshow('Smart MSL Translator',temp_frame)
            cv2.waitKey(1)
            
            # tukar actual logical ayat
            final_output =get_llama_sentence(sentence_list)
            sentence_list=[] 
            is_thinking=True 

    #draw the normal user interface box
    cv2.rectangle(frame,(0, 420),(640,480),(30,30,30),-1)
    
    if is_thinking==True:
        color=(0,255,0)
    else:
        color=(255,255,255)
        
    cv2.putText(frame,final_output,(20,460), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    cv2.imshow('Smart MSL Translator',frame)
    
    #check keyboard keys
    if cv2.waitKey(1) == ord('q'): 
        break
    if cv2.waitKey(1) == ord('c'):
        sentence_list=[]
        final_output="Cleared."
        is_thinking=False
cap.release()
cv2.destroyAllWindows()