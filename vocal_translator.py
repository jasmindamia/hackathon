import cv2
import pickle
import numpy as np
import collections
import time
import os
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS 
import pygame 
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- 1. INITIALIZATION ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pygame.mixer.init() # Initialize the audio engine

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error: Model file not found. {e}")
    exit()

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Variables
history = collections.deque(maxlen=11)
sentence_list = []
last_word, counter = "", 0
final_output = "Sistem Sedia..."
last_hand_time = time.time()
is_thinking = False

def speak_text(text):
    """Converts Malay text to audio and plays it immediately."""
    try:
        # Stop any audio that is currently playing to avoid file lock errors
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        
        tts = gTTS(text=text, lang='ms') # 'ms' for Bahasa Melayu
        filename = "temp_voice.mp3"
        tts.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Audio Error: {e}")

def get_llama_translation(words):
    """Uses Llama 3.1 to turn keywords into formal Malay."""
    if not words: return ""
    raw_input = " ".join(words)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional MSL interpreter. Convert raw sign language keywords into one natural, formal Malay sentence. Return ONLY the sentence. make it not more than 60 characters"},
                {"role": "user", "content": f"Keywords: {raw_input}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.6
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return raw_input.capitalize()

print("Level 6 Active. Vocal MSL Translator Ready.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    current_coords = [0.0] * 84
    
    if results.multi_hand_landmarks:
        last_hand_time = time.time() 
        is_thinking = False
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            lbl = results.multi_handedness[i].classification[0].label
            idx = 0 if lbl == "Right" else 42
            coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
            current_coords[idx:idx+42] = coords
        
        history.append(current_coords)

        if len(history) == 11:
            motion_window = history[0] + history[5] + history[10]
            word = model.predict([np.asarray(motion_window)])[0]

            if word == last_word:
                counter += 1
            else:
                counter, last_word = 0, word

            if counter >= 18:
                if not sentence_list or word != sentence_list[-1]:
                    sentence_list.append(word)
                    final_output = "Input: " + " ".join(sentence_list)
                counter = 0
    else:
        # --- 2.5 SECOND SILENCE GAP TRIGGER ---
        silence_gap = time.time() - last_hand_time
        if silence_gap > 2.5 and len(sentence_list) > 0 and not is_thinking:
            # 1. Update UI to show thinking
            final_output = "AI Menjana Suara..."
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (0, 420), (640, 480), (0, 0, 0), -1)
            cv2.putText(temp_frame, final_output, (20, 460), 1, 1.2, (0, 255, 255), 2)
            cv2.imshow('Smart MSL Translator', temp_frame)
            cv2.waitKey(1)
            
            # 2. Get Llama Translation
            smart_result = get_llama_translation(sentence_list)
            final_output = smart_result
            
            # 3. Speak the result
            speak_text(smart_result)
            
            sentence_list = [] 
            is_thinking = True 

    # --- UI DISPLAY ---
    cv2.rectangle(frame, (0, 420), (640, 480), (35, 35, 35), -1)
    text_color = (0, 255, 0) if is_thinking else (255, 255, 255)
    cv2.putText(frame, final_output, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    cv2.imshow('Smart MSL Translator', frame)
    
    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        sentence_list, final_output, is_thinking = [], "Cleared.", False
        pygame.mixer.music.stop()

cap.release()
cv2.destroyAllWindows()