import os
import cv2
import numpy as np
from gtuner import *
import gtuner
import time
from random import randrange

'''
<version>1.0</version>

<shortdesc>
Returnal : Setup Line Guide for Auto Overload "returnal_auto_overload.py"
<i>Tested @ 1920x1080 input using PS Remote Play</i>
</shortdesc>

<keywords>Returnal, Computervision, CV, Auto, Auto Fire, AutoFire</keywords>

<donate>N/A</donate>
<docurl>N/A</docurl>
'''

class GCVWorker:
    def __init__(self, width, height):
        self.gcvdata = bytearray([0xFF])
        self.offsetX = 0
        self.offsetY = 0
        self.contour2 = np.array([[120 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[120 + self.offsetX, 13 + self.offsetY]])
        self.contour3 = np.array([[120 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[120 + self.offsetX, 41 + self.offsetY]])

        self.contours = [self.contour2, self.contour3]
        self.interval = 0
        self.toggle = False

    def process(self, frame):

        if self.interval % 30 == 0:
            self.interval = 0
            self.toggle = not self.toggle
        framebak = frame.copy()
        self.interval = self.interval + 1

        frame = cv2.circle(frame, (117 + self.offsetX, 13 + self.offsetY), 1, (20, 20, 220), 1)
        frame = cv2.circle(frame, (207 + self.offsetX, 13 + self.offsetY), 1, (20, 20, 220), 1)
        frame = cv2.circle(frame, (117 + self.offsetX, 42 + self.offsetY), 1, (20, 20, 220), 1)
        frame = cv2.circle(frame, (207 + self.offsetX, 42 + self.offsetY), 1, (20, 20, 220), 1)

        cv2.putText(frame, "Offset X: " + str(self.offsetX), (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Offset Y: " + str(self.offsetY), (2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        for i, contour in enumerate(self.contours):
            x,y,w,h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255, 0.25),1)

        return (frame if self.toggle else framebak, None)

    def __del__(self):
        del self.gcvdata
        del self.offsetX
        del self.offsetY
        del self.contour2
        del self.contour3
        del self.contours
        del self.interval
        del self.toggle
