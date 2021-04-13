from PIL import Image, ImageTk
import tkinter as tk
import h5py
import numpy as np
from cv2 import cv2
import mediapipe as mp
from os.path import exists
from time import sleep
from keras.models import load_model
from keyboard import press_and_release as press

filename = 'dynamic_gestures2.hdf5'
if not exists(filename):
    f = h5py.File(filename, 'x')
else:
    f = h5py.File(filename, 'r+')
current_dataset = None

# How long a gesture is in frames.
gestureLength = 20
framesRemaining = gestureLength
gestureAnchor = (0, 0, 0)

# 20 Most Recent Frames
latestFrames = []
# Used Model
model = load_model('test_model_2')
model_labels = ["flick left", "flick right", "point down", "point left", "point right", "point up", "swipe down", "swipe up", "swipe right"]
keys = {
    "point up":"i",
    "point left":'j',
    "point down":'k',
    "point right":'l',
    "flick left":"b",
    "flick right":"b"
}
def toggleRecording():
    global recording, startStopButton, current_dataset, framesRemaining, gestureLength
    recording = not recording
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
    elif framesRemaining <= 0:
        framesRemaining = gestureLength
        startStopButton.config(text="Start Recording")


def show_frame():
    global cap, current_dataset, framesRemaining, gestureLength, gestureAnchor, latestFrames, predictionString, currentGesture, guessToggle, keys 
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
        if guessToggle.get():
            # Only collect frames if we have under gesture length
            if hand != [] and len(latestFrames) < gestureLength:
                latestFrames = list(latestFrames)
                latestFrames.append(np.array(hand))
                latestFrames = np.array(latestFrames)
            # Remove first frame and add new frame to end then recenter everything
            elif hand != []:
                latestFrames = list(latestFrames)
                latestFrames.append(np.array(hand))
                latestFrames = np.array(latestFrames[-gestureLength:])
                tempAnchor = [latestFrames[0][0][i] for i in range(3)]
                for timestep in range(gestureLength):
                    for point in range(21):
                        for coord in range(3):
                            latestFrames[timestep][point][coord] -= tempAnchor[coord]
                # Used for predicting current gesture.
                prediction = model.predict(latestFrames[None,:])
                modelGuess = model_labels[np.argmax(prediction)]
                predictionString.set("Current Guess: " + modelGuess + " " + str(np.round(np.max(prediction)*100, decimals=2))+"%")
                if currentGesture != modelGuess:
                    if currentGesture != None:
                        press(keys[currentGesture])
                        currentGesture = None
                        predictionString.set("No Gesture Detected.")
                        latestFrames = np.array([])
                        sleep(.25)
                    else: currentGesture = modelGuess
        else:
            currentGesture = None
            predictionString.set("No Gesture Detected.")
            latestFrames = np.array([])
        # If collecting data, add data to dataset
        if recording:
            if framesRemaining <= 0:
                toggleRecording()
            # Add our data to h5 file.
            current_dataset[current_dataset.shape[0]-1,
                            gestureLength-framesRemaining, :, :] = np.array(hand)
            framesRemaining -= 1
    # Remove frames one by one as no gesture is being done
    elif guessToggle.get():
        predictionString.set("No Gesture Detected.")
        latestFrames = list(latestFrames)
        if currentGesture != None:
                press(keys[currentGesture])
                currentGesture = None
        if len(latestFrames) > 1: latestFrames = latestFrames[1:]
        elif len(latestFrames) == 1: latestFrames = []
        latestFrames = np.array(latestFrames)
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
gestureEntryLabel.grid(row=0, column=1)
gestureNameEntry = tk.Entry(ctrlPanel)
gestureNameEntry.grid(row=0, column=2, padx=10)
startStopButton = tk.Button(
    ctrlPanel, text="Start Recording", command=toggleRecording)
startStopButton.grid(row=0, column=3, padx=10)

# Guess of Gesture
guessToggle = tk.IntVar() 
currentGesture = None
guessToggleButton = tk.Checkbutton(ctrlPanel, text="Guess?", variable=guessToggle, onvalue=True, offvalue=False)
#guessToggleButton.pack()
guessToggleButton.grid(row=0, column=0, padx=5)
predictionString = tk.StringVar()
predictionString.set("No Guess.")
currentLabel = tk.Label(ctrlPanel, textvariable=predictionString)
currentLabel.grid(row=0, column=4)

# Setup MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6, min_tracking_confidence=0.75, max_num_hands=1
)

# OpenCV video stream at 20 fps.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 20) 
show_frame()
window.mainloop()
hands.close()
cap.release()