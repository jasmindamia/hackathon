import customtkinter as ctk
import webbrowser
import cv2
import pickle
import numpy as np
import collections
import time
import os
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageTk
import mediapipe as mp
import pygame
from gtts import gTTS

# --- 1. INITIALIZATION ---
load_dotenv()
my_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=my_key)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Audio for Level 6
pygame.mixer.init()

# --- 2. GLOBAL AI FUNCTIONS ---

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an MSL translator. Convert raw keywords into a formal, natural Malay sentence. Return ONLY the sentence. Example: 'SAYA SEKOLAH' -> 'Saya mahu pergi ke sekolah.'. Make it not more than 60 characters"
}

def get_llama_sentence(words):
    """Level 5: AI Sentence Builder"""
    if not words: return " "
    raw_input = " ".join(words)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                SYSTEM_PROMPT,
                {"role": "user", "content": f"Keywords: {raw_input}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.5
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print("Llama API Error:", e)
        return raw_input.capitalize()

def speak_text(text):
    """Level 6: Voice Assistance"""
    try:
        # Stop any existing audio
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        
        tts = gTTS(text=text, lang='ms')
        filename = "temp_voice.mp3"
        tts.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Voice Error: {e}")

# --- 3. MAIN APPLICATION CLASS ---

class SenseAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sense-AI: Inclusion in Motion")
        self.geometry("1100x850")
        
        # Container to swap pages
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        
        self.help_url = "www.bimsignbank.org/groups/daily-life/language" # Replace with actual link
        self.show_page("menu")

    def show_page(self, page_name):
        for widget in self.container.winfo_children():
            widget.destroy()

        if page_name == "menu":
            page = MainMenu(self.container, 
                            learn_cmd=lambda: self.show_page("learn"),
                            trans_cmd=lambda: self.show_page("translate"),
                            help_url=self.help_url)
        elif page_name == "learn":
            # You can apply a similar structure to your Learn tab
            page = LearnPage(self.container, back_cmd=lambda: self.show_page("menu"))
        elif page_name == "translate":
            page = TranslatePage(self.container, back_cmd=lambda: self.show_page("menu"))

        page.pack(fill="both", expand=True)

# --- 4. PAGE CLASSES ---

class MainMenu(ctk.CTkFrame):
    def __init__(self, master, learn_cmd, trans_cmd, help_url):
        super().__init__(master)
        ctk.CTkLabel(self, text="SENSE-AI", font=("Roboto", 50, "bold")).pack(pady=(80, 10))
        ctk.CTkLabel(self, text="Inclusion in Motion", font=("Roboto", 20)).pack(pady=(0, 50))
        
        ctk.CTkButton(self, text="1. LEARN (Single Sign)", width=350, height=50, fg_color="#3498db", command=learn_cmd).pack(pady=10)
        ctk.CTkButton(self, text="2. TRANSLATE (Sentence Builder)", width=350, height=50, fg_color="#2ecc71", command=trans_cmd).pack(pady=10)
        ctk.CTkButton(self, text="3. HELP (MSL Library)", width=350, height=50, fg_color="#f39c12", command=lambda: webbrowser.open(help_url)).pack(pady=10)
        ctk.CTkButton(self, text="4. EXIT", width=350, height=50, fg_color="#e74c3c", command=master.master.quit).pack(pady=10)

class TranslatePage(ctk.CTkFrame):
    def __init__(self, master, back_cmd):
        super().__init__(master)
        
        # Load Model
        try:
            with open('./model.p', 'rb') as f:
                self.model = pickle.load(f)['model']
        except:
            self.model = None

        # Logic Variables (From your original script)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.history_list = []
        self.sentence_list = []
        self.last_word = ""
        self.counter = 0
        self.final_output = "Llama System Ready..."
        self.last_hand_time = time.time()
        self.is_thinking = False

        # UI Layout
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)
        
        self.output_label = ctk.CTkLabel(self, text=self.final_output, font=("Roboto", 24, "bold"), wraplength=800)
        self.output_label.pack(pady=20)

        # Control Panel
        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(pady=10)
        
        # Level 6 Voice Toggle
        self.voice_toggle = ctk.CTkSwitch(self.ctrl_frame, text="Voice Output (Level 6)")
        self.voice_toggle.grid(row=0, column=0, padx=10)
        self.voice_toggle.select()

        ctk.CTkButton(self.ctrl_frame, text="Clear", width=120, command=self.clear_all).grid(row=0, column=1, padx=10)
        ctk.CTkButton(self.ctrl_frame, text="Back", width=120, command=lambda: self.stop_camera(back_cmd)).grid(row=0, column=2, padx=10)

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def clear_all(self):
        self.sentence_list = []
        self.final_output = "Cleared."
        self.is_thinking = False
        self.output_label.configure(text=self.final_output)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Use your exact current_coords logic
            current_coords = [0.0] * 84
            
            if results.multi_hand_landmarks:
                self.last_hand_time = time.time()
                self.is_thinking = False

                for i in range(len(results.multi_hand_landmarks)):
                    hand_lms = results.multi_hand_landmarks[i]
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    lbl = results.multi_handedness[i].classification[0].label
                    idx = 0 if lbl == "Right" else 42
                    
                    coords = []
                    for lm in hand_lms.landmark:
                        coords.append(lm.x)
                        coords.append(lm.y)
                    
                    for position in range(42):
                        current_coords[idx+position] = coords[position]
                
                self.history_list.append(current_coords)
                if len(self.history_list) > 11:
                    self.history_list.pop(0)

                if len(self.history_list) == 11 and self.model:
                    motion_window = self.history_list[0] + self.history_list[5] + self.history_list[10]
                    word = self.model.predict([np.asarray(motion_window)])[0]

                    if word == self.last_word:
                        self.counter += 1
                    else:
                        self.counter = 0
                        self.last_word = word

                    if self.counter >= 18:
                        if not self.sentence_list or word != self.sentence_list[-1]:
                            self.sentence_list.append(word)
                        self.counter = 0
                        
                        all_words = " ".join(self.sentence_list)
                        self.final_output = "Input: " + all_words
                        self.output_label.configure(text=self.final_output, text_color="white")
            else:
                # Silence gap timeout
                gap = time.time() - self.last_hand_time
                if gap > 2.5 and self.sentence_list and not self.is_thinking:
                    self.is_thinking = True
                    self.output_label.configure(text="Llama is thinking...", text_color="#f1c40f")
                    
                    # Call Llama
                    self.final_output = get_llama_sentence(self.sentence_list)
                    self.output_label.configure(text=self.final_output, text_color="#2ecc71")
                    
                    # Call Voice if toggled
                    if self.voice_toggle.get() == 1:
                        speak_text(self.final_output)
                    
                    self.sentence_list = [] # Reset for next sentence

            # Update Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def stop_camera(self, cmd):
        self.cap.release()
        cmd()

# Placeholder for LearnPage
class LearnPage(ctk.CTkFrame):
    def __init__(self, master, back_cmd):
        super().__init__(master)
        
        # Setup AI Tools
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.history = collections.deque(maxlen=11)
        
        try:
            with open('./model.p', 'rb') as f:
                self.model = pickle.load(f)['model']
        except:
            self.model = None
            print("Model not found!")

        # UI SETUP
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)
        
        self.result_label = ctk.CTkLabel(self, text="Waiting for sign...", font=("Roboto", 24, "bold"))
        self.result_label.pack(pady=20)

        ctk.CTkButton(self, text="Back to Menu", command=lambda: self.stop_camera(back_cmd)).pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # --- AI LOGIC ---
            current_coords = [0.0] * 84
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    # Draw Skeleton
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract Coordinates
                    lbl = results.multi_handedness[i].classification[0].label
                    idx = 0 if lbl == "Right" else 42
                    coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
                    current_coords[idx:idx+42] = coords
                
                self.history.append(current_coords)
                
                if len(self.history) == 11 and self.model:
                    motion_window = self.history[0] + self.history[5] + self.history[10]
                    prediction = self.model.predict([np.asarray(motion_window)])[0]
                    self.result_label.configure(text=f"Detected: {prediction}", text_color="#3498db")

            # Update Video Frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def stop_camera(self, back_cmd):
        self.cap.release()
        back_cmd()

if __name__ == "__main__":
    app = SenseAIApp()
    app.mainloop()
