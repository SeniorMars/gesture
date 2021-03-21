from PIL import Image, ImageTk
import tkinter as tk
import h5py
import numpy as np
from cv2 import cv2
import mediapipe as mp
from os.path import exists
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


filename = 'dgestures.hdf5'
if not exists(filename):
    f = h5py.File(filename, 'x')
else:
    f = h5py.File(filename, 'r+')
current_dataset = None

model = tf.keras.models.load_model('test_model_1')
model_labels = ["five_finger", "four_finger", "ok",
                "one_finger", "three_finger", "two_finger"]


def toggleRecording():
    global recording, startStopButton, current_dataset
    recording = not recording
    if recording:
        startStopButton.config(text="Stop Recording")
        # Open up the corresponding data set for the gesture type
        gestureType = gestureNameEntry.get()
        if "/"+gestureType+"/ds1" not in f:
            current_dataset = f.create_dataset(
                '/'+gestureType+'/ds1', (0, 21, 3), maxshape=(None, 21, 3), dtype='float64')
        else:
            current_dataset = f[gestureType]['ds1']
    else:
        startStopButton.config(text="Start Recording")


def show_frame():
    global cap, current_dataset, predictionString
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
        wrist = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
        wrist = (wrist.x, wrist.y, wrist.z)
        hand = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.ListFields()[0][1]:
                # Add each point to the hand array, normalized relative to the anchor.
                hand.append(tuple(np.subtract((i.x, i.y, i.z), wrist)))
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if recording:
            # Extends h5 dataset, and add our data to it.
            current_dataset.resize(
                (current_dataset.shape[0]+1, current_dataset.shape[1], current_dataset.shape[2]))
            current_dataset[current_dataset.shape[0]-1, :, :] = np.array(hand)
        # Used for predicting current gesture.
        prediction = model.predict(np.array(hand)[None, :])
        predictionString.set("Current Guess: " + model_labels[np.argmax(
            prediction)] + " " + str(np.round(np.max(prediction)*100, decimals=2))+"%")

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

predictionString = tk.StringVar()
predictionString.set("No Guess.")
currentLabel = tk.Label(ctrlPanel, textvariable=predictionString)
currentLabel.grid(row=0, column=4)

# Setup MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.75, max_num_hands=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20)
show_frame()
window.mainloop()
hands.close()
cap.release()
