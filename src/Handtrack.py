import numpy as np
from cv2 import cv2
import mediapipe as mp

import h5py

import tkinter as tk
from PIL import Image, ImageTk


def toggleRecording():
    global recording, recordStatus, startStopButton
    recording = not recording
    recordStatus.config(text="Recording: " + str(recording))
    if recording:
        startStopButton.config(text="Stop Recording")
    else:
        startStopButton.config(text="Start Recording")


def show_frame():
    global cap
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        return

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
        wrist = (wrist.x, wrist.y, wrist.z)
        hand = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.ListFields()[0][1]:
                hand.append(tuple(np.subtract((i.x, i.y, i.z), wrist)))
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


# Window setup
window = tk.Tk()
window.wm_title("Gesture Recognizer")
window.config(background="#FFFFFF")
# Video Stream Frame
imgFrame = tk.Frame(window, width=600, height=600)
imgFrame.grid(row=0, column=0, padx=10, pady=2)
lmain = tk.Label(imgFrame)
lmain.grid(row=0, column=0)
# Recording buttons interface
recording = False
recordStatus = tk.Label(window, text="Recording: " + str(recording))
recordStatus.grid(row=0, column=1)
startStopButton = tk.Button(window, text="Start Recording", command=toggleRecording)
startStopButton.grid(row=1, column=1)
# Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row=600, column=0, padx=10, pady=2)

# Setup MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1
)

cap = cv2.VideoCapture(0)
show_frame()
window.mainloop()
hands.close()
cap.release()
