import numpy as np
import cv2

#Open webcam stream
stream = cv2.VideoCapture(0)

while(True):
    # Capture frame from stream
    ret, frame = stream.read()

    # Show frame
    cv2.imshow('frame',frame)
    #Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()