import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from picamera2 import Picamera2
import time

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def get_eye_landmarks(face_landmarks):
    left_eye = np.array([(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)])
    right_eye = np.array([(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)])
    return left_eye, right_eye

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
current_time = time.monotonic()
EAR=0.3
while True:
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cur_time = time.monotonic()
    if cur_time - current_time > 1:
        current_time = cur_time
        faces = hog_face_detector(gray)
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)

            left_eye, right_eye = get_eye_landmarks(face_landmarks)

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)

            EAR = (left_ear + right_ear) / 2
            EAR = round(EAR, 2)
    if EAR < 0.26:
        #oci su zatvorene, do something
    print(EAR)

    cv2.imshow("CE keyboard", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

picam2.close()
cv2.destroyAllWindows()