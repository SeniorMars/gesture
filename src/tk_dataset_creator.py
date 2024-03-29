import tkinter as tk
from PIL import Image, ImageTk
from tk_control_panel import ControlPanel
from tk_data_recorder import DatasetRecorder

from cv2 import cv2
import numpy as np
import mediapipe as mp

TARGET_FRAMERATE: int = 20


class DatasetCreator(tk.Tk):
    """
    Main Window
    """

    def __init__(self, *args, **kwargs):
        # TKinter setup
        tk.Tk.__init__(self, *args, **kwargs)
        self.wm_title("Dataset Creator")

        # MediaPipe setup
        self.mpHands = mp.solutions.hands.Hands(
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

        self.controlPanel = ControlPanel(self, toggleCommand=self.onToggleClicked)
        self.controlPanel.grid(row=1, column=0)

        # Data Recording utility
        self.recorder: DatasetRecorder = DatasetRecorder("gesture_dataset.h5", 20)

        # Start event loop
        self.appLoop()

    def onToggleClicked(self) -> None:
        self.recorder.setCurrentGesture(self.controlPanel.gestureNameInput.value())

    def appLoop(self) -> None:
        """
        Event loop
        """
        _, hand = self.fetchHand()

        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA))
        imgtk = ImageTk.PhotoImage(image=img)
        self.videoLabel.imgtk = imgtk
        self.videoLabel.configure(image=imgtk)
        self.videoLabel.after(int(1000 / TARGET_FRAMERATE), self.appLoop)

        if self.controlPanel.isRecording():
            self.recorder.addFrameToSample(hand)
            if self.recorder.currentFrame == self.recorder.sampleLength:
                self.recorder.addSampleToDataset()
                self.controlPanel.recordingToggle.toggle()

    def fetchHand(self, drawHand=True) -> tuple:
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
        results = self.mpHands.process(self.image)
        # Draw the hand annotations on the image.
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = np.array(
                    [(i.x, i.y, i.z) for i in hand_landmarks.ListFields()[0][1]]
                )
                if drawHand:
                    mp.solutions.drawing_utils.draw_landmarks(
                        self.image,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )
                return (True, hand)
        return (False, None)


if __name__ == "__main__":
    app = DatasetCreator()
    app.mainloop()
