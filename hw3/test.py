import numpy as np
import cv2
import matplotlib.pyplot as plt

#img = cv2.imread('/home/koo/homework/hw3/tt.JPG')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#plt.imshow(img, cmap='gray')
#plt.show()


import cv2

cap = cv2.VideoCapture("home/koo/homework/hw3/tt.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("home/koo/homework/hw3/tt.mp4")
    cv2.waitKey(1000)
    print "Wait for the header"

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
while True:
    flag, frame = cap.read()
    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        print str(pos_frame)+" frames"
    else:
        # The next frame is not ready, so we try to read it again
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print "frame is not ready"
        # It is better to wait for a while for the next frame to be ready
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
