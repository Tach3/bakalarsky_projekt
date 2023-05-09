import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from picamera2 import Picamera2
import time
import random
import tkinter as tk
from PIL import Image, ImageTk

"""
Tkinter funkcie
"""
def check_input(event, word):
    if input_field.get() == word:
        print("You wrote " + word +" correctly")
        app.quit()
    else:
        print("You did not write " + word + " correctly")
        app.quit()
"""
Both
"""
ear_value = 0

def calculate_ear():
    global ear_value
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ear_value = are_closed(hog_face_detector(gray), dlib_facelandmark, gray)
    label_widget.after(2000, calculate_ear)

def open_camera():
  
    # Capture the video frame by frame
    frame = picam2.capture_array()
    global ear_value
  
    # Get the current EAR value
    if ear_value is None:
        current_ear = 5
    else:
        current_ear = ear_value
    
    if current_ear > 0.26:
        input_field.config(state="disabled") # Disable input field
        cv2.putText(frame,"CHEATER",(20,100),
            cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        cv2.putText(frame,"Close your eyes",(20,400),
            cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        print("open eyes")
    else:
        input_field.config(state="normal")
    print(current_ear)

    # Convert image from one color space to other
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
  
    # Capture the latest frame and transform to image
    captured_image = Image.fromarray(opencv_image)
  
    # Convert captured image to photoimage
    photo_image = ImageTk.PhotoImage(image=captured_image)
  
    # Displaying photoimage in the label
    label_widget.photo_image = photo_image
  
    # Configure image in the label
    label_widget.configure(image=photo_image)
  
    # Repeat the same process after every 10 seconds
    label_widget.after(100, open_camera)

"""
OpenCV Funkcie
"""
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def extract_eye_coords(face_landmarks):
    left_eye_coords = []
    right_eye_coords = []
    for n in range(36, 42):
        x, y = face_landmarks.part(n).x, face_landmarks.part(n).y
        left_eye_coords.append((x, y))
    for n in range(42, 48):
        x, y = face_landmarks.part(n).x, face_landmarks.part(n).y
        right_eye_coords.append((x, y))
    return np.array(left_eye_coords), np.array(right_eye_coords)

def are_closed(faces, dlib, gray):
    for face in faces:

            face_landmarks = dlib(gray, face)
            leftEye, rightEye = extract_eye_coords(face_landmarks)
            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)

            return EAR
"""
quality of life funkcie
"""
def pick_word():
    with open('words.txt', 'r') as f:
        words = f.read().splitlines()
    return random.choice(words)


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

#detectors
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#nacita nahodne slovo z txt suboru
word = pick_word()

# Create a GUI app
app = tk.Tk()

app.title("Blind Writing")

# Bind the app with Escape keyboard to
# quit app whenever pressed
app.bind('<Escape>', lambda e: app.quit())

label_widget = tk.Label(app, text="Write this: " + word)
label_widget.pack()

button1 = tk.Button(app, text="Start", command=open_camera)
button1.pack()
label_widget.after(2000, calculate_ear)

input_field = tk.Entry(app)
input_field.pack()   
input_field.bind("<Return>", lambda event: check_input(event, word))
app.mainloop()
