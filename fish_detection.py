import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture('./video/f4k_detection_tracking/gt_122.flv')
kernel = np.ones((5, 5), np.uint8)

backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
while True:
    ret, frame = video.read()
    if not ret:
        break
    fgmask = backgroundObject.apply(frame)
    # _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    # To get rid of the noise
    fgmast = cv2.erode(fgmask, kernel, iterations = 1)
    fgmask = cv2.dilate(fgmask, kernel, iterations = 2)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()
    for cnt in contours:
      if cv2.contourArea(cnt) > 250:
        x,y,width,height = cv2.boundingRect(cnt)
        cv2.rectangle(frameCopy,(x,y),(x+width,y+height),(0,0,255),2)
        cv2.putText(frameCopy,'Fish',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1,cv2.LINE_AA)
    foregroundPart = cv2.bitwise_and(frame, frame, mask = fgmask)
    stacked = np.hstack((frame,frameCopy))
    cv2.imshow('Original Frame, Extracted Frame, Detected Fishes',cv2.resize(stacked, None, fx = 0.8, fy = 0.8))
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()