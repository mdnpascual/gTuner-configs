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
Returnal : Auto Overload / Fast Reload - CV Companion script for "Returnal.gpc"
Computer Vision : Detects if reloading/out of ammo and press R2 at the right time to fast reload
<i>Tested @ 1920x1080 input using PS Remote Play</i>
</shortdesc>

<keywords>Returnal, Computervision, CV, Auto, Auto Fire, AutoFire</keywords>

<donate>N/A</donate>
<docurl>N/A</docurl>
'''

class GCVWorker:
    def __init__(self, width, height):
        self.gcvdata = bytearray([0xFF])
        self.triggerColorBoundaryV1 = [ #RGB Boundary
            [180,155,91], [235,200,150]
        ]
        self.overloadConfidence = 0.50
        self.offsetX = 0
        self.offsetY = 0
        self.contour2 = np.array([[120 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[120 + self.offsetX, 13 + self.offsetY]])
        self.contour3 = np.array([[120 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[120 + self.offsetX, 41 + self.offsetY]])

        self.contours = [self.contour2, self.contour3]
        self.interval = 0
        self.toggle = False
        self.debug = False
        self.drawGuide = True

    def process(self, frame):

        if self.debug:
            if self.interval % 30 == 0:
                self.interval = 0
                self.toggle = not self.toggle
            framebak = frame.copy()
            self.interval = self.interval + 1

        detected = []

        for i, contour in enumerate(self.contours):
            x,y,w,h = cv2.boundingRect(contour)
            detected.append(np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(np.uint8))

        frame, isOverloadReady = self.isOverloadReadyV2(frame, detected)

        if isOverloadReady:
            cv2.putText(frame, "Firing!", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            if self.drawGuide:
                for i, contour in enumerate(self.contours):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0, 0.25),1)
            self.gcvdata[0] = 2
            return (frame, self.gcvdata)
        else:
            cv2.putText(frame, "Scanning...", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            if self.drawGuide:
                for i, contour in enumerate(self.contours):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255, 0.25),1)
        if self.debug:
            return (frame if self.toggle else framebak, None)
        else:
            return (frame, None)

    def __del__(self):
        del self.gcvdata
        del self.triggerColorBoundaryV1
        del self.overloadConfidence
        del self.offsetX
        del self.offsetY
        del self.contour2
        del self.contour3
        del self.contours
        del self.interval
        del self.toggle
        del self.debug
        del self.drawGuide

    def isOverloadReadyV2(self, frame, result):
        valToReturn = True
        for i, boundary in enumerate(result):
            # if self.debug:
                # print("isOverloadReadyV2 BGR: ", boundary)
            valToReturn = valToReturn and self.isColorInRangeOfReadyV1(boundary)

        if valToReturn and self.debug:
            print("isOverloadReadyV2 triggered firing")
        if not valToReturn:
            frame, valToReturn = self.isOverloadReadyV1(frame)
        if valToReturn and self.debug:
            print("isOverloadReadyV1 triggered firing")
        return frame, valToReturn

    def isOverloadReadyV1(self, frame):
        rand1 = randrange(18, 105)
        rand2 = randrange(130, 198)
        rand3 = randrange(130, 198)
        ptsToCheck = (frame[13 + self.offsetY,117 + self.offsetX], frame[13 + self.offsetY,207 + self.offsetX], frame[42 + self.offsetY, 117 + self.offsetX], frame[42 + self.offsetY, 207 + self.offsetX], frame[27 + self.offsetY, rand1 + self.offsetX], frame[13 + self.offsetY, rand2 + self.offsetX], frame[42 + self.offsetY, rand3 + self.offsetX])
        results = map(self.isColorInRangeOfReadyV1, ptsToCheck)
        results = list(results)
        resultSum = sum(result for result in results)
        # if(resultSum > 1 and self.debug):
        #     print(results)
        #     print(list(ptsToCheck))
        valToReturn = resultSum / 7 > self.overloadConfidence
        if self.drawGuide:
            frame = cv2.circle(frame, (117 + self.offsetX, 13 + self.offsetY), 1, (20, 220 if results[0] else 20, 20 if results[0] else 220), 1)
            frame = cv2.circle(frame, (207 + self.offsetX, 13 + self.offsetY), 1, (20, 220 if results[1] else 20, 20 if results[1] else 220), 1)
            frame = cv2.circle(frame, (117 + self.offsetX, 42 + self.offsetY), 1, (20, 220 if results[2] else 20, 20 if results[2] else 220), 1)
            frame = cv2.circle(frame, (207 + self.offsetX, 42 + self.offsetY), 1, (20, 220 if results[3] else 20, 20 if results[3] else 220), 1)
            frame = cv2.circle(frame, (rand1 + self.offsetX, 27 + self.offsetY), 1, (20, 220 if results[4] else 20, 20 if results[4] else 220), 1)
            frame = cv2.circle(frame, (rand2 + self.offsetX, 13 + self.offsetY), 1, (20, 220 if results[5] else 20, 20 if results[5] else 220), 1)
            frame = cv2.circle(frame, (rand3 + self.offsetX, 42 + self.offsetY), 1, (20, 220 if results[6] else 20, 20 if results[6] else 220), 1)
        return frame, valToReturn

    def isColorInRangeOfReadyV1(self, bgr):
        result = False
        if len(bgr) == 4:
            b,g,r,a = (bgr)
        else:
            b,g,r = (bgr)

        if(b >  self.triggerColorBoundaryV1[0][2] and b <  self.triggerColorBoundaryV1[1][2]):#bluecheck
            if(g >  self.triggerColorBoundaryV1[0][1] and g <  self.triggerColorBoundaryV1[1][1]):#greencheck
                if(r >  self.triggerColorBoundaryV1[0][0] and r <  self.triggerColorBoundaryV1[1][0]):#redcheck
                    return True

        return result
