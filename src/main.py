#!/usr/bin/env python3
import numpy as np
from cv2 import cv2


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, img)


# Open webcam stream
stream = cv2.VideoCapture(0)
while True:
    # Capture frame from stream
    ret, frame = stream.read()
    showInMovedWindow("original", frame, 0, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    showInMovedWindow("Hue", frame[:, :, 0], 600, 0)
    showInMovedWindow("Saturation", frame[:, :, 1], 1200, 0)
    showInMovedWindow("Value", frame[:, :, 2], 1800, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    sat = frame[:, :, 1]
    sat = cv2.erode(sat, kernel, iterations=3)
    # (_, sat) = cv2.threshold(sat, 300, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sat = cv2.dilate(sat, kernel, iterations=3)
    mask = cv2.bitwise_and(sat, frame[:, :, 2], mask=sat)
    # Show frame
    showInMovedWindow("proc_frame", sat, 0, 600)
    showInMovedWindow("mask", mask, 0, 1200)
    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()
