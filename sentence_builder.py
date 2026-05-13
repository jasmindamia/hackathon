import cv2
import pickle
import numpy as np
import collections
import time
import os
from dotenv import load_dotenv
from groq import Groq # Switched from google.generativeai
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# --- 1. CONFIGURATION ---
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# System instruction to keep Llama focused
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an MSL translator. Convert raw keywords into a formal, natural Malay sentence. Return ONLY the sentence. Example: 'SAYA SEKOLAH' -> 'Saya mahu pergi ke sekolah.'"
}

try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Error: model.p not found!")
    exit()

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Variables
history = collections.deque(maxlen=11)
sentence_list = []
last_word, counter = "", 0
final_output = "Llama System Ready..."
last_hand_time = time.time()
is_thinking = False

def get_llama_sentence(words):
    """Uses Llama 3 to refine the grammar."""
    if not words: return ""
    raw_input = " ".join(words)
    try:
        # Using Llama 3 70B or 8B (8B is faster for demos)
        chat_completion = client.chat.completions.create(
            messages=[
                SYSTEM_PROMPT,
                {"role": "user", "content": f"Keywords: {raw_input}"}
            ],
            model="llama3-8b-8192", 
            temperature=0.5
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Llama API Error: {e}")
        return raw_input.capitalize()

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
            if word == last_word: counter += 1
            else: counter, last_word = 0, word

            if counter >= 18:
                if not sentence_list or word != sentence_list[-1]:
                    sentence_list.append(word)
                counter = 0
                final_output = "Input: " + " ".join(sentence_list)
    else:
        # Timeout Logic
        silence_gap = time.time() - last_hand_time
        if silence_gap > 2.5 and len(sentence_list) > 0 and not is_thinking:
            final_output = "Llama is thinking..."
            
            # Temporary UI update
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (0, 420), (640, 480), (0, 0, 0), -1)
            cv2.putText(temp_frame, final_output, (20, 460), 1, 1.2, (0, 255, 255), 2)
            cv2.imshow('Smart MSL Translator', temp_frame)
            cv2.waitKey(1)
            
            final_output = get_llama_sentence(sentence_list)
            sentence_list = [] 
            is_thinking = True 

    # UI Rendering
    cv2.rectangle(frame, (0, 420), (640, 480), (30, 30, 30), -1)
    color = (0, 255, 0) if is_thinking else (255, 255, 255)
    cv2.putText(frame, final_output, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow('Smart MSL Translator', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        sentence_list, final_output, is_thinking = [], "Cleared.", False

cap.release()
cv2.destroyAllWindows()