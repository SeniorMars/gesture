import tkinter as tk
from PIL import Image, ImageTk

from cv2 import cv2
import numpy as np
import mediapipe as mp

from keyboard import press_and_release as press
from json import load

from data_preprocessor import DataGenerator
from gestures import GESTURES


import tensorflow as tf

TARGET_FRAMERATE: int = 20
GESTURE_LENGTH: int = 20
TFLITE_MODEL_PATH: str = "saved_models/MODEL-2021-06-02-16-12-10.tflite"
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

keys = load(open("keybinds.json", "r"))
for key in keys:
    if key in GESTURES:
        GESTURES[key]['keybind'] = keys[key]


class LiveModelTester(tk.Tk):
    """
    Main Window
    """

    def __init__(self, *args, **kwargs):
        # TKinter setup
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("Gesture Recognition Tester")

        # MediaPipe setup
        self.mpHands = mp.solutions.hands.Hands(
            min_detection_confidence=0.6, min_tracking_confidence=0.75, max_num_hands=1
        )
        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        # OpenCV current frame
        self.image = None

        # Video Stream Frame
        self.videoFrame = tk.Frame(self, width=800, height=800)
        self.videoFrame.grid(row=0, column=0, padx=10, pady=10)
        self.videoLabel = tk.Label(self.videoFrame)
        self.videoLabel.grid(row=0, column=0)

        self.predictionLabel = tk.Label(self, text="")
        self.predictionLabel.grid(row=1, column=0)

        # Toggle keyboard input
        self.keyboardToggle = tk.BooleanVar()
        self.useKeyboardToggle = tk.Checkbutton(
            self, text="Send Keypresses", onvalue=True, offvalue=False, variable=self.keyboardToggle)
        self.useKeyboardToggle.grid(row=1, column=1, padx=5)

        self.frameCache = []
        self.skipFrames = 0

        self.interpreter = tf.lite.Interpreter(TFLITE_MODEL_PATH)
        self.interpreter.allocate_tensors()

        # Start event loop
        self.appLoop()

    def appLoop(self) -> None:
        """
        Event loop
        """
        success, hand = self.fetchHand()
        if success and self.skipFrames <= 0:
            self.frameCache.append(hand)
            if len(self.frameCache) > GESTURE_LENGTH:
                self.frameCache.pop(0)
            self.updatePrediction()
        self.skipFrames -= 1

        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA))
        imgtk = ImageTk.PhotoImage(image=img)
        self.videoLabel.imgtk = imgtk
        self.videoLabel.configure(image=imgtk)
        self.videoLabel.after(int(1000 / TARGET_FRAMERATE), self.appLoop)

    def updatePrediction(self):
        if len(self.frameCache) != GESTURE_LENGTH:
            return
        sample = np.array(
            DataGenerator.center_sample(np.array(self.frameCache))[None, :],
            dtype="float32",
        )

        self.interpreter.set_tensor(
            self.interpreter.get_input_details()[0]["index"], sample
        )
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(
            self.interpreter.get_output_details()[0]["index"]
        )

        gestureLabel = str(list(GESTURES)[np.argmax(prediction)])
        gestureCertainty = round(np.max(prediction) * 100, 2)
        predictionString = "{} {}%".format(gestureLabel, str(gestureCertainty))
        self.predictionLabel.config(text=predictionString)

        if self.keyboardToggle.get() and gestureCertainty > 96 and "keybind" in GESTURES[gestureLabel]:
            press(GESTURES[gestureLabel]['keybind'])
            # print(gestureLabel)
            # empty framecache
            self.frameCache = []
            self.skipFrames = 10

    def fetchHand(self, draw_hand=True) -> tuple:
        """
        Returns a tuple of (success, hand), where hand is
        a Hand is an array of shape (21,3)

        Also sets self.image property to a frame
        with the hand drawn on it.
        """
        success, self.image = self.cap.read()
        if not success:
            return (False, None)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        self.image = cv2.cvtColor(cv2.flip(self.image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        self.image.flags.writeable = False
        results = self.mpHands.process(self.image)
        # Draw the hand annotations on the image.
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = np.array(
                    [(i.x, i.y, i.z)
                     for i in hand_landmarks.ListFields()[0][1]]
                )
                if draw_hand:
                    mp.solutions.drawing_utils.draw_landmarks(
                        self.image,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )
                return (True, hand)
        return (False, None)


if __name__ == "__main__":
    app = LiveModelTester()
    app.mainloop()
