import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import pickle
import collections

# --- MEDIA PIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SenseAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sense-AI: Inclusion in Motion")
        self.geometry("1000x800")
        
        # Load your trained model (Level 3 output)
        try:
            model_dict = pickle.load(open('./model.p', 'rb'))
            self.trained_model = model_dict['model']
        except:
            self.trained_model = None
            print("Warning: model.p not found. Prediction will not work.")

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)
        self.show_page("menu")

    def show_page(self, page_name):
        for widget in self.container.winfo_children():
            widget.destroy()

        if page_name == "menu":
            page = MainMenu(self.container, start_command=lambda: self.show_page("translator"))
        elif page_name == "translator":
            page = TranslatorPage(self.container, 
                                  back_command=lambda: self.show_page("menu"),
                                  model=self.trained_model)
        page.pack(fill="both", expand=True)

# --- PAGE 1: MAIN MENU (Help Button Removed) ---
class MainMenu(ctk.CTkFrame):
    def __init__(self, master, start_command):
        super().__init__(master)
        
        ctk.CTkLabel(self, text="SENSE-AI", font=("Roboto", 50, "bold")).pack(pady=(100, 20))
        ctk.CTkLabel(self, text="Inclusion in Motion", font=("Roboto", 20)).pack(pady=(0, 50))
        
        # FIXED: Changed 'size=(300, 50)' to 'width=300, height=50'
        ctk.CTkButton(self, text="START TRANSLATOR", width=300, height=50, corner_radius=10,
                      fg_color="#2ecc71", hover_color="#27ae60",
                      command=start_command).pack(pady=10)
        
        # FIXED: Same here for the exit button
        ctk.CTkButton(self, text="EXIT SYSTEM", width=300, height=50, corner_radius=10,
                      fg_color="#e74c3c", hover_color="#c0392b",
                      command=master.quit).pack(pady=10)
# --- PAGE 2: THE TRANSLATOR HUB ---
class TranslatorPage(ctk.CTkFrame):
    def __init__(self, master, back_command, model):
        super().__init__(master)
        self.model = model
        
        # MediaPipe Setup inside the page
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.history = collections.deque(maxlen=11)
        self.sentence_list = []
        
        # UI Elements
        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)
        
        self.output_text = ctk.CTkLabel(self, text="Waiting for hand signs...", 
                                       font=("Roboto", 24, "bold"), text_color="#3498db")
        self.output_text.pack(pady=20)
        
        self.back_btn = ctk.CTkButton(self, text="BACK TO MENU", command=self.stop_and_back)
        self.back_btn.pack(pady=10)

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. FIX: Mirror the camera
            frame = cv2.flip(frame, 1)
            
            # 2. FIX: Process AI Logic (Levels 1 & 4)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            current_coords = [0.0] * 84
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    # Draw the skeleton onto the frame
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Process coordinates for prediction
                    lbl = results.multi_handedness[i].classification[0].label
                    idx = 0 if lbl == "Right" else 42
                    coords = [c for lm in hand_lms.landmark for c in (lm.x, lm.y)]
                    current_coords[idx:idx+42] = coords
                
                self.history.append(current_coords)
                
                # Predict if we have enough history
                if len(self.history) == 11 and self.model:
                    motion_window = self.history[0] + self.history[5] + self.history[10]
                    prediction = self.model.predict([np.asarray(motion_window)])[0]
                    self.output_text.configure(text=f"Detected: {prediction}")

            # 3. Convert frame for CustomTkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Loop the function
        self.after(10, self.update_frame)

    def stop_and_back(self):
        self.cap.release()
        # Call the back command provided by the app
        self.master.master.show_page("menu") 

if __name__ == "__main__":
    app = SenseAIApp()
    app.mainloop()