from PIL import Image, ImageTk
import tkinter as tk
import h5py
import numpy as np
from cv2 import cv2
import mediapipe as mp
from os.path import exists


filename = 'dynamic_gestures.hdf5'
if not exists(filename):
    f = h5py.File(filename, 'x')
else:
    f = h5py.File(filename, 'r+')
current_dataset = None

# How long a gesture is in frames.
gestureLength = 20
framesRemaining = gestureLength
gestureAnchor = (0, 0, 0)


def toggleRecording():
    global recording, startStopButton, current_dataset, framesRemaining, gestureLength
    recording = not recording
    framesRemaining = gestureLength
    if recording:
        startStopButton.config(text="Stop Recording")
        # Open up the corresponding data set for the gesture type
        gestureType = gestureNameEntry.get()
        if "/"+gestureType not in f:
            current_dataset = f.create_dataset(
                '/'+gestureType, (1, gestureLength, 21, 3), maxshape=(None, gestureLength, 21, 3), dtype='float64')
        else:
            current_dataset = f[gestureType]
            current_dataset.resize(
                (current_dataset.shape[0]+1, current_dataset.shape[1], current_dataset.shape[2], current_dataset.shape[3]))
    else:
        startStopButton.config(text="Start Recording")


def show_frame():
    global cap, current_dataset, framesRemaining, gestureLength, gestureAnchor
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
        # Get the wrist point of the hand to use as an anchor.
        if framesRemaining == gestureLength:
            wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
            gestureAnchor = (wrist.x, wrist.y, wrist.z)
        hand = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.ListFields()[0][1]:
                # Add each point to the hand array, normalized relative to the anchor.
                hand.append(tuple(np.subtract((i.x, i.y, i.z), gestureAnchor)))
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if recording:
            if framesRemaining <= 0:
                toggleRecording()
            # Add our data to h5 file.
            current_dataset[current_dataset.shape[0]-1,
                            gestureLength-framesRemaining, :, :] = np.array(hand)
            framesRemaining -= 1

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

# Control Panel Setup
ctrlPanel = tk.Frame(window)
ctrlPanel.grid(row=1, column=0)

# Recording buttons interface
recording = False
gestureEntryLabel = tk.Label(ctrlPanel, text="Gesture Name:")
gestureEntryLabel.grid(row=0, column=0)
gestureNameEntry = tk.Entry(ctrlPanel)
gestureNameEntry.grid(row=0, column=1, padx=10)
startStopButton = tk.Button(
    ctrlPanel, text="Start Recording", command=toggleRecording)
startStopButton.grid(row=0, column=2, padx=10)

# Setup MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.75, max_num_hands=1
)

# OpenCV video stream at 20 fps.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
show_frame()
window.mainloop()
hands.close()
cap.release()