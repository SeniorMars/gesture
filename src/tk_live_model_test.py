from keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk

from cv2 import cv2
import numpy as np
import mediapipe as mp

from data_preprocessor import DataGenerator, GESTURES

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices[0]) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

TARGET_FRAMERATE: int = 20


class LiveModelTester(tk.Tk):
    """
    Main Window 
    """

    def __init__(self, *args, **kwargs):
        # TKinter setup
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("Gesture Recognition Tester")

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.6, min_tracking_confidence=0.75, max_num_hands=1
        )
        # OpenCV setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FRAMERATE)

        # OpenCV current frame
        self.image = None

        # Video Stream Frame
        self.videoFrame = tk.Frame(self, width=800, height=800)
        self.videoFrame.grid(row=0, column=0, padx=10, pady=10)
        self.videoLabel = tk.Label(self.videoFrame)
        self.videoLabel.grid(row=0, column=0)

        self.predictionLabel = tk.Label(self, text="")
        self.predictionLabel.grid(row=1, column=0)

        self.frameCache = []
        self.model = load_model('saved_models\MODEL-2021-05-24-21-27-14')

        # Start event loop
        self.appLoop()

    def appLoop(self) -> None:
        """
        Event loop
        """
        success, hand = self.fetchHand()
        if success:
            self.frameCache.append(hand)
            if len(self.frameCache) > 20:
                self.frameCache.pop(0)

        self.updatePrediction()

        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA))
        imgtk = ImageTk.PhotoImage(image=img)
        self.videoLabel.imgtk = imgtk
        self.videoLabel.configure(image=imgtk)
        self.videoLabel.after(int(1000/TARGET_FRAMERATE), self.appLoop)

    def updatePrediction(self):
        if len(self.frameCache) != 20:
            return
        sample = DataGenerator.center_sample(
            np.array(self.frameCache))[None, :]
        prediction = self.model.predict(sample)
        gestureLabel = str(list(GESTURES)[np.argmax(prediction)])
        gestureCertainty = str(round(np.max(prediction)*100,2))
        predictionString = "{} {}%".format(gestureLabel, gestureCertainty)
        self.predictionLabel.config(text=predictionString)

    def fetchHand(self, draw_hand=True) -> tuple:
        """
        Returns a tuple of (success, hand), where hand is
        a Hand is an array of shape (20,21,3)

        Also sets this object's image property to a frame
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
        results = self.mp_hands.process(self.image)
        # Draw the hand annotations on the image.
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = np.array([(i.x, i.y, i.z)
                                 for i in hand_landmarks.ListFields()[0][1]])
                if draw_hand:
                    mp.solutions.drawing_utils.draw_landmarks(
                        self.image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    )
                return (True, hand)
        return (False, None)


if __name__ == '__main__':
    app = LiveModelTester()
    app.mainloop()