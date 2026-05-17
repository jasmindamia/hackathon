import os
import time
import pickle
import numpy as np
import collections
import cv2
import pygame
import customtkinter as ctk
import webbrowser
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageTk
import mediapipe as mp
from gtts import gTTS

load_dotenv()
apikey = os.getenv("GROQ_API_KEY")
aiclient = Groq(api_key=apikey)

mediapipehands = mp.solutions.hands
drawingtools = mp.solutions.drawing_utils
handtracker = mediapipehands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

pygame.mixer.init()

guideurl = "http://www.bimsignbank.org/groups/daily-life/language"

trackinghistory = []
sentencewords = []
previouslydetectedword = ""
stabilitycounter = 0
counter = 0 
lastseenhandtime = time.time()
isprocessing = False
videocapture = None

def normalize_hand_coordinates(raw_coords):
    if sum(raw_coords) == 0.0:
        return list(raw_coords)

    wrist_x = raw_coords[0]
    wrist_y = raw_coords[1]

    shifted_coords = []
    for i in range(0, len(raw_coords), 2):
        shifted_coords.append(raw_coords[i] - wrist_x)
        shifted_coords.append(raw_coords[i+1] - wrist_y)

    max_value = max(max(abs(val) for val in shifted_coords), 1e-6)
    return [val / max_value for val in shifted_coords]

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are an MSL translator. Convert raw keywords into a formal, natural Malay sentence. Return ONLY the sentence. Example: 'SAYA SEKOLAH' -> 'Saya mahu pergi ke sekolah.'. Make it not more than 60 characters"
}

def getllamasentence(words):
    if not words: 
        return " "
    raw_input = " ".join(words)
    try:
        chatcompletion = aiclient.chat.completions.create(
            messages=[
                SYSTEM_PROMPT,
                {"role": "user", "content": f"Keywords: {raw_input}"}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.5
        )
        return chatcompletion.choices[0].message.content.strip()
    except Exception as e:
        print("Llama API Code Error:", e)
        return raw_input

def speaktext(texttospeak):
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        
        texttospeech = gTTS(text=texttospeak, lang='ms')
        temporaryfilename = "temp_voice.mp3"
        texttospeech.save(temporaryfilename)
        
        pygame.mixer.music.load(temporaryfilename)
        pygame.mixer.music.play()
    except Exception as e:
        print("Audio playback error:", e)

class SenseAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sense-AI: Inclusion in Motion")
        self.geometry("1100x650")
        
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        
        self.help_url = guideurl
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
            page = learnpage(self.container, back_cmd=lambda: self.show_page("menu"))
        elif page_name == "translate":
            page = translatepage(self.container, back_cmd=lambda: self.show_page("menu"))

        page.pack(fill="both", expand=True)

class MainMenu(ctk.CTkFrame):
    def __init__(self, master, learn_cmd, trans_cmd, help_url):
        super().__init__(master, fg_color="#1a1a2e") 
        ctk.CTkLabel(self, text="SENSE-AI", font=("Segoe UI", 28, "bold"), text_color="#a0aec0").pack(pady=(60, 5))
        ctk.CTkLabel(self, text="VOICES BEYOND SOUND", font=("Segoe UI Semibold", 42), text_color="white").pack(pady=(0, 40))
        
        ctk.CTkButton(self, text="LEARN with MSL!", width=350, height=45, fg_color="#3498db", hover_color="#2980b9", command=learn_cmd).pack(pady=10)
        ctk.CTkButton(self, text="TRANSLATE", width=350, height=45, fg_color="#2ecc71", hover_color="#27ae60", command=trans_cmd).pack(pady=10)
        ctk.CTkButton(self, text="GUIDE (MSL Link)", width=350, height=45, fg_color="#f39c12", hover_color="#d35400", command=lambda: webbrowser.open(help_url)).pack(pady=10)
        ctk.CTkButton(self, text="EXIT", width=350, height=45, fg_color="#e74c3c", hover_color="#c0392b", command=master.master.quit).pack(pady=10)

def clearwindow():
    for item in rootWindow.winfo_children():
        item.destroy()

class learnpage(ctk.CTkFrame):
    def __init__(self, master, back_cmd):
        super().__init__(master, fg_color="#1a1a2e") 
        
        self.hands = mediapipehands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.history = collections.deque(maxlen=11)
        
        try:
            with open('./model.p', 'rb') as f:
                self.model = pickle.load(f)['model']
        except:
            self.model = None
            print("Model configuration file 'model.p' missing inside directory.")

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)
        
        self.result_label = ctk.CTkLabel(self, text="Waiting for sign...", font=("Roboto", 24, "bold"), text_color="#f1c40f")
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
            
            current_coords = [0.0] * 84
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    drawingtools.draw_landmarks(frame, hand_lms, mediapipehands.HAND_CONNECTIONS)
                    
                    lbl = results.multi_handedness[i].classification[0].label
                    idx = 0 if lbl == "Right" else 42
                    coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
                    
                    normalized_coords = normalize_hand_coordinates(coords)
                    current_coords[idx:idx+42] = normalized_coords
                
                self.history.append(current_coords)
                
                if len(self.history) == 11 and self.model:
                    motion_window = self.history[0] + self.history[5] + self.history[10]
                    prediction = self.model.predict([np.asarray(motion_window)])[0]
                    self.result_label.configure(text=f"Detected: {prediction}", text_color="#3498db")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def stop_camera(self, back_cmd):
        self.cap.release()
        back_cmd()

class translatepage(ctk.CTkFrame):
    def __init__(self, master, back_cmd):
        super().__init__(master, fg_color="#1a1a2e") 
        
        try:
            with open('./model.p', 'rb') as f:
                self.model = pickle.load(f)['model']
        except:
            self.model = None

        self.hands = mediapipehands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.history_list = []
        self.sentence_list = []
        self.last_word = ""
        self.counter = 0
        
        self.final_output = "Llama System Ready..."
        self.last_hand_time = time.time()
        self.is_thinking = False

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)
        
        self.output_label = ctk.CTkLabel(self, text=self.final_output, font=("Roboto", 24, "bold"), text_color="#d35400", wraplength=800)
        self.output_label.pack(pady=20)

        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(pady=10)
        
        self.voice_toggle = ctk.CTkSwitch(self.ctrl_frame, text="Voice Output ", text_color="#f1c40f")
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
            
            current_coords = [0.0] * 84
            
            if results.multi_hand_landmarks:
                self.last_hand_time = time.time()
                self.is_thinking = False

                for i in range(len(results.multi_hand_landmarks)):
                    hand_lms = results.multi_hand_landmarks[i]
                    drawingtools.draw_landmarks(frame, hand_lms, mediapipehands.HAND_CONNECTIONS)
                    
                    lbl = results.multi_handedness[i].classification[0].label
                    idx = 0 if lbl == "Right" else 42
                    
                    coords = []
                    for lm in hand_lms.landmark:
                        coords.append(lm.x)
                        coords.append(lm.y)
                    
                    normalized_coords = normalize_hand_coordinates(coords)
                    for position in range(42):
                        current_coords[idx+position] = normalized_coords[position]
                
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
                gap = time.time() - self.last_hand_time
                if gap > 2.5 and self.sentence_list and not self.is_thinking:
                    self.is_thinking = True
                    self.output_label.configure(text="Llama is thinking...", text_color="#f1c40f")
                    self.update_idletasks() 
                    
                    self.final_output = getllamasentence(self.sentence_list)
                    self.output_label.configure(text=self.final_output, text_color="#2ecc71")
                    
                    if self.voice_toggle.get() == 1:
                        speaktext(self.final_output)
                    
                    self.sentence_list = []

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    def stop_camera(self, cmd):
        self.cap.release()
        cmd()

if __name__ == "__main__":
    app = SenseAIApp()
    app.mainloop()