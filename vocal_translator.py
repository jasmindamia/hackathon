import cv2
import pickle
import numpy as np
import time
import os
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS 
import pygame 
import mediapipe as mp

#initialisation
load_dotenv()
my_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=my_key)

#audio
pygame.mixer.init() 

#buka .model,p
try:
    file=open('./model.p', 'rb')
    model_dict=pickle.load(file)
    file.close()
    model=model_dict['model']
except:
    print("Error: Model file not found.")
    exit()

#setup mediapipe
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

#open cam
cap=cv2.VideoCapture(0)

#variables
history_list=[]
sentence_list=[]
last_word=""
counter=0
final_output="Sistem Sedia..."
last_hand_time=time.time()
is_thinking=False

def speak_text(text):
    try:
        #stop any audio that is currently playing
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        
        #make the speech audio file using Bahasa Melayu
        tts=gTTS(text=text,lang='ms') 
        filename="temp_voice.mp3"
        tts.save(filename)
        
        #play the audio file out loud
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print("Audio Error:", e)

def get_llama_translation(words):
    if len(words)==0: 
        return ""
        
    #leaving spce between words
    raw_input = ""
    for w in words:
        raw_input = raw_input + w + " "
    raw_input = raw_input.strip()
    
    try:
        #ask llama model to make a nice sentence
        chat_completion=client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional MSL interpreter. Convert raw sign language keywords into one natural, formal Malay sentence. Return ONLY the sentence. make it not more than 60 characters"},
                {"role": "user", "content": "Keywords: " + raw_input}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.6
        )
        answer=chat_completion.choices[0].message.content
        return answer.strip()
    except Exception as e:
        return raw_input.capitalize()

print("Level 6 Active. Vocal MSL Translator Ready.")

while True:
    success, frame=cap.read()
    if success==False: 
        break
        
    frame=cv2.flip(frame,1)
    
    #change color frame for mediapipe processing
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    #initialize 84 zeros inside list manually
    current_coords=[]
    for zero in range(84):
        current_coords.append(0.0)
    
    if results.multi_hand_landmarks !=None:
        last_hand_time=time.time() 
        is_thinking=False
        
        hand_index=0
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_lms,mp_hands.HAND_CONNECTIONS)
            
            lbl=results.multi_handedness[hand_index].classification[0].label
            
            if lbl=="Right":
                idx=0
            else: 
                idx=42
                
            #clear and collect landmarks
            coords=[]
            for lm in hand_lms.landmark:
                coords.append(lm.x)
                coords.append(lm.y)
                
            #copy collected coords into our main coordinate list
            for position in range(42):
                current_coords[idx + position] = coords[position]
                
            hand_index = hand_index + 1
        
        #record historical hand movements
        history_list.append(current_coords)
        
        #drop old frames to keep history at 11 frames maximum (0.3s]
        if len(history_list)>11:
            history_list.pop(0)

        if len(history_list)==11:
            motion_window=history_list[0]+history_list[5]+history_list[10]
            
            array_data=np.asarray(motion_window)
            prediction=model.predict([array_data])
            word=prediction[0]

            if word==last_word:
                counter=counter+1
            else:
                counter=0
                last_word=word

            if counter>=18:
                #verify word isn't a duplicate of our previous detected sign
                if len(sentence_list) == 0 or word != sentence_list[-1]:
                    sentence_list.append(word)
                    
                    #build display text manually out of sentence list items
                    all_words = ""
                    for w in sentence_list:
                        all_words=all_words+w+" "
                    final_output="Input: "+all_words.strip()
                    
                counter=0
    else:
        #2.5second after the hand disappears,ai akan start buat kerja
        silence_gap=time.time()-last_hand_time
        if silence_gap > 2.5 and len(sentence_list)>0 and is_thinking==False:
            
            #update visual window layout to let user know it's processing
            final_output="AI Menjana Suara..."
            temp_frame=frame.copy()
            cv2.rectangle(temp_frame, (0,420), (640,480), (0,0,0),-1)
            cv2.putText(temp_frame,final_output, (20,460),1,1.2, (0,255,255),2)
            cv2.imshow('Smart MSL Translator',temp_frame)
            cv2.waitKey(1)
            
            #translate
            smart_result = get_llama_translation(sentence_list)
            final_output = smart_result
            
            #audio output
            speak_text(smart_result)
            
            sentence_list = [] 
            is_thinking = True 

    #screen display
    cv2.rectangle(frame, (0,420), (640,480), (35,35,35),-1)
    
    if is_thinking==True:
        text_color=(0,255,0)
    else:
        text_color=(255,255,255)
        
    cv2.putText(frame,final_output, (20,460), cv2.FONT_HERSHEY_SIMPLEX,0.6,text_color,2)
    cv2.imshow('Smart MSL Translator',frame)
    
    #quit n clear
    key = cv2.waitKey(1)
    
    if key == ord('q'): 
        break
        
    if key == ord('c'):
        sentence_list = []
        final_output = "Cleared."
        is_thinking = False
        pygame.mixer.music.stop()

cap.release()
cv2.destroyAllWindows()
