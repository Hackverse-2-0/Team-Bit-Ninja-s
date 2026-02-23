import cv2
import mediapipe as mp
import threading
import requests
import speech_recognition as sr
import google.generativeai as genai
import time
import io
import PIL.Image
import numpy as np

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
PI_IP = "http://192.168.137.229:5000" 

# IMPORTANT: Keep your API key safe. Use environment variables in a real project.
GEMINI_API_KEY = "AIzaSyBVJfIvUmJrc3kEM2ajq3FdhOOOMleYQu8"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

chat_session = model.start_chat(history=[
    {"role": "user", "parts": "You are Ally, a medical assistant robot. Keep responses concise (1-2 sentences) and strictly medical. Do not use asterisks or markdown formatting."}
])

recognizer = sr.Recognizer()

# ==========================================
# 🗣️ COMMUNICATION & VISION FUNCTIONS (Laptop to Pi)
# ==========================================
def speak(text, block_mic=True):
    """Sends a text command to the Pi to be spoken."""
    print(f"\n🗣️ Ally: {text}")
    try: 
        requests.post(f"{PI_IP}/receive_instruction", json={"text": text}, timeout=3)
        if block_mic:
            time.sleep(max(2.0, len(text.split()) * 0.35)) 
    except requests.exceptions.RequestException: 
        print("❌ Error: Connection to Ally's body lost.")

def get_vitals():
    """Gets the latest sensor data from the Pi."""
    try:
        return requests.get(f"{PI_IP}/get_health_data", timeout=2).json()
    except:
        return None

def process_vision(prompt, text):
    """Grabs a frame from the Pi and sends it to Gemini for analysis."""
    try:
        res = requests.get(f"{PI_IP}/capture", timeout=5)
        if res.status_code == 200:
            print("🧠 Processing visual data...")
            img = PIL.Image.open(io.BytesIO(res.content))
            full_prompt = prompt + f" The user asked: '{text}'"
            response = model.generate_content([full_prompt, img])
            speak(response.text.replace('*', '').replace('#', ''))
        else:
            speak("My optical sensor is currently obstructed.")
    except Exception:
        speak("My vision processing unit experienced a timeout.")

def ask_gemini_text(prompt):
    """Sends a text-only prompt to the Gemini chat session."""
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception:
        return "I am having trouble accessing my neural network."

# ==========================================
# 👁️ LIVE BONE TRACKING & AI COACH
# ==========================================
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def start_live_vision_coach():
    """This function now contains the ENTIRE live video and coaching loop."""
    print("📺 Activating Live Vision Coach...")
    speak("Activating physio-vision. I will now monitor your movements. Press Q on the video window to stop.", block_mic=False)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(f"{PI_IP}/video_feed")
    
    if not cap.isOpened():
        print(f"❌ Could not connect to the video feed at {PI_IP}/video_feed.")
        speak("I'm having trouble connecting to my camera feed.")
        return

    last_feedback_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                
                cv2.putText(frame, str(int(angle)), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                current_time = time.time()
                if current_time - last_feedback_time > 6.0:
                    feedback_text = None
                    if angle < 40: feedback_text = "Elbow is bent. Good. Now extend fully."
                    elif angle > 160: feedback_text = "Arm fully extended. Perfect."
                    
                    if feedback_text:
                        threading.Thread(target=speak, args=(feedback_text, False)).start()
                        last_feedback_time = current_time
            except Exception:
                pass
        
        cv2.imshow('MedBot Live AI Coach', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Deactivating physio-vision.", block_mic=False)
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Live Vision Coach deactivated.")

# ==========================================
# 🧠 AI AUDIO LISTENER (The main loop of the program)
# ==========================================
def run_ai_brain():
    print("\n" + "="*50)
    print("🧠 ALLY AI BRAIN ONLINE")
    print(f"Connected to Med-Bot at {PI_IP}")
    print("="*50 + "\n")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        speak("All systems nominal. AI core online.", block_mic=True)
        
        while True:
            try:
                print("🎤 Listening for command...")
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=8)
                text = recognizer.recognize_google(audio).lower()
                
                if len(text.split()) < 2 and text not in ["hello", "hi", "hey", "stop"]: 
                    continue

                print(f"🧑 You: {text}")

                # --- CORRECTED IF/ELIF LOGIC ---
                if any(w in text for w in ["coach", "exercise", "physio", "posture", "back pain", "movement"]):
                    start_live_vision_coach()
                
                elif any(w in text for w in ["hello", "hi", "hey"]):
                     speak("Hello! I am Ally. How can I assist you today?")

                elif any(w in text for w in ["check my health", "check my vitals"]):
                    speak("Please place your finger on the sensor.")
                    time.sleep(3) 
                    vitals = get_vitals()
                    if vitals:
                        speak(f"Your heart rate is {vitals['heart_rate']} BPM. Oxygen is {vitals['spo2']} percent.")

                elif any(w in text for w in ["medicine", "pill", "drug"]):
                    speak("Please hold the medicine clearly in front of my camera.")
                    process_vision("Identify this medication. State its purpose and 1 common side effect.", text)

                elif "stop" in text:
                    speak("Powering down modules.")
                    break
                
                # *THIS IS THE CRITICAL FIX*: This 'else' block now correctly catches
                # any phrase that didn't match the specific keywords above.
                else:
                    speak(ask_gemini_text(f"Respond concisely to: '{text}'"))

            except sr.UnknownValueError:
                print("... (Could not understand audio)")
                continue
            except Exception as e:
                print(f"An error occurred in the brain loop: {e}")
                pass 

if _name_ == "_main_":
    run_ai_brain()
