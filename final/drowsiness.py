import cv2
import dlib
from scipy.spatial import distance
import numpy as np
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
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ear_value = are_closed(hog_face_detector(gray), dlib_facelandmark, gray)
    label_widget.after(1000, calculate_ear)

def open_camera():
  
    # Capture the video frame by frame
    _, frame = cap.read()
    global ear_value
  
    # Get the current EAR value
    if ear_value is None:
        current_ear = 5
    else:
        current_ear = ear_value
    if current_ear > 0.26:
        input_field.config(state="disabled")
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
    label_widget.after(10, open_camera)

"""
OpenCV Funkcie
"""

def pick_word():
    with open('words.txt', 'r') as f:
        words = f.read().splitlines()
    return random.choice(words)

def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

def extract_eye_coords(face_landmarks):
    left_eye = np.array([(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)])
    right_eye = np.array([(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)])
    return left_eye, right_eye

def are_closed(faces, dlib, gray):
    for face in faces:

            face_landmarks = dlib(gray, face)
            leftEye, rightEye = extract_eye_coords(face_landmarks)
            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)

            return EAR

cap = cv2.VideoCapture(0)
# Declare the width and height in variables


#detectors
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
label_widget.after(1000, calculate_ear)

input_field = tk.Entry(app)
input_field.pack()   
input_field.bind("<Return>", lambda event: check_input(event, word))
#label_widget.after(500, cheater_punish)
app.mainloop()

# while(time.time() - start_time) < 4:   
#     _, fr = cap.read()
#     cv2.putText(fr,"Write this: ",(20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
#     cv2.putText(fr,word,(20,400),
#                 cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
#     cv2.imshow("Keyboard & Closed Eyes", fr)
#     cv2.waitKey(1)


# current_time = time.monotonic()
# EAR=0.3
# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cur_time = time.monotonic()
#     if cur_time - current_time > 1:
#         current_time = cur_time        
        


            
    

#     cv2.imshow("Keyboard & Closed Eyes", frame)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()
