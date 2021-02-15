import numpy as np
import cv2

#Open webcam stream
stream = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()
while(True):
    # Capture frame from stream
    ret, frame = stream.read()
    mask = backSub.apply(frame)
    cv2.dilate(mask,(3,3))

    # Show frame
    cv2.imshow('frame',mask)
    #Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()