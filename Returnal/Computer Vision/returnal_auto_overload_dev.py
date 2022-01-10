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

<keywords>Returnal, Computervision, CV, Auto, </keywords>

<donate>N/A</donate>
<docurl>N/A</docurl>
'''

class GCVWorker:
    def __init__(self, width, height):
        self.gcvdata = bytearray([0xFF])
        self.width = width
        self.height = height
        self.piConst = 3.141592
        # self.triggerColorBoundaryV1 = [ #RGB Boundary
        #     [180,155,91], [235,200,150]
        # ]
        # self.inhibitColorBoundaryV1 = [ #RGB Boundary
        #     [150,180,190], [205,210,210]
        # ]
        self.triggerColorBoundaryV1 = [ #RGB Boundary
            [180,155,91], [235,200,150]
        ]
        self.inhibitColorBoundaryV1 = [ #RGB Boundary
            [140,150,135], [180,180,170]
        ]
        self.waitReference = [[181, 202, 161], [181, 202, 161]]
        self.waitBoundary = [50, 40, 40]
        self.overloadConfidence = 0.50
        self.earlyOverloadConfidence = 0.50
        # self.offsetX = 4
        # self.offsetY = 0
        self.offsetX = 0
        self.offsetY = 0
        self.offsetBlendY = 5
        # self.contour1 = np.array([[4 + self.offsetX, 23 + self.offsetY],[7 + self.offsetX, 23 + self.offsetY],[7 + self.offsetX, 31 + self.offsetY],[4 + self.offsetX, 31 + self.offsetY]])
        self.contour2 = np.array([[120 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[204 + self.offsetX, 13 + self.offsetY],[120 + self.offsetX, 13 + self.offsetY]])
        self.contour3 = np.array([[120 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[204 + self.offsetX, 41 + self.offsetY],[120 + self.offsetX, 41 + self.offsetY]])

        self.contours = [self.contour2, self.contour3]
        if (width % 16 == 0) and (height % 9 == 0):
            print("Note: Only tested using 16/9 ratio display")
        self.interval = 0
        self.toggle = False
        self.debug = False
        self.drawGuide = True

    def process(self, frame):

        # frame = cv2.drawContours(frame, [self.contour1], 0, (255,255,255), 1)
        # frame = cv2.imread('test/waitFail.png')

        if self.debug:
            print("--------START--------")
            if self.interval % 30 == 0:
                self.interval = 0
                self.toggle = not self.toggle
            framebak = frame.copy()
            self.interval = self.interval + 1

        detected = []
        correction = []

        for i, contour in enumerate(self.contours):
            x,y,w,h = cv2.boundingRect(contour)

            # frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0, 0.25),1)
            detected.append(np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(np.uint8))
            correction.append(np.array(cv2.mean(frame[y+self.offsetBlendY:y+self.offsetBlendY+h,x:x+w])).astype(np.uint8))
            # print(i, ': Average color (BGR): ',np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(np.uint8))

        frame, isOverloadReady = self.isOverloadReadyV2(frame, detected)
        # frame, isOverloadEarly = self.isOverloadEarlyV2(frame, detected, correction)
        isOverloadEarly = False

        # frame, isOverloadReady = self.isOverloadReadyV1(frame)
        # frame, isOverloadEarly = self.isOverloadEarlyV1(frame)

        if self.debug:
            print("--------END--------")

        if isOverloadEarly:
            cv2.putText(frame, "Inhibiting Fire", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            if self.drawGuide:
                for i, contour in enumerate(self.contours):
                    x,y,w,h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0, 0.25),1)
            self.gcvdata[0] = 1
            return (frame, self.gcvdata)
        elif isOverloadReady:
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
        del self.width
        del self.height
        del self.piConst
        del self.triggerColorBoundaryV1
        del self.inhibitColorBoundaryV1
        del self.overloadConfidence
        del self.earlyOverloadConfidence
        del self.offsetX
        del self.offsetY
        # del self.contour1
        del self.contour2
        del self.contour3
        del self.contours

    def lineDetect(self, frame):
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, self.piConst/180, 60, None, 50, 5)

        for line in lines:
            for x1, y1, x2, y2 in line:
                frame = cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 20), 1)

        return frame

    def isOverloadEarlyV2(self, frame, detected, correction):
        valToReturn = True
        for i, boundary in enumerate(detected):
            if self.debug:
                print("isOverloadEarlyV2 BGR: ", boundary)
            valToReturn = valToReturn and self.isColorInRangeOfEarlyV1(boundary)

        if valToReturn and self.debug:
            print("isColorInRangeOfEarlyV1 triggered inhibit")
            return frame, valToReturn
        if not valToReturn:
            valToReturn = self.isOverlayEarlyCorrection(frame, detected, correction)

        if valToReturn and self.debug:
            print("isOverlayEarlyCorrection triggered inhibit")
            return frame, valToReturn
        if not valToReturn:
            frame, valToReturn = self.isOverloadEarlyV1(frame)

        if valToReturn and self.debug:
            print("isOverloadEarlyV1 triggered inhibit")
            return frame, valToReturn

        return frame, valToReturn

    def isOverlayEarlyCorrection(self, frame, detected, correction):
        screened = []
        for i in range(2):
            color = []
            for j in range(3):
                color.append(int(( 1 - (1 - (self.waitReference[i][j] / 255) ) * (1 - (correction[i][j] / 255) )) * 255))

            color.append(0)
            screened.append(color)

        return self.isCorrectionInRangeOfEarlyV1(detected, screened)

    def isOverloadReadyV2(self, frame, result):
        valToReturn = True
        for i, boundary in enumerate(result):
            if self.debug:
                print("isOverloadReadyV2 BGR: ", boundary)
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

    def isOverloadEarlyV1(self, frame):
        rand2 = randrange(130, 198)
        rand3 = randrange(130, 198)
        ptsToCheck = (frame[13 + self.offsetY,117 + self.offsetX], frame[13 + self.offsetY,207 + self.offsetX], frame[42 + self.offsetY, 117 + self.offsetX], frame[42 + self.offsetY, 207 + self.offsetX], frame[13 + self.offsetY, rand2 + self.offsetX], frame[42 + self.offsetY, rand3 + self.offsetX])
        results = map(self.isColorInRangeOfEarlyV1, ptsToCheck)
        results = list(results)
        resultSum = sum(result for result in results)
        # if(resultSum > 1 and self.debug):
        #     print(results)
        #     print(list(ptsToCheck))
        valToReturn = resultSum / 6 > self.earlyOverloadConfidence
        if valToReturn and False:
            frame = cv2.circle(frame, (117 + self.offsetX, 13 + self.offsetY), 1, (20 if results[0] else 220, 220 if results[0] else 20, 20), 1)
            frame = cv2.circle(frame, (207 + self.offsetX, 13 + self.offsetY), 1, (20 if results[1] else 220, 220 if results[1] else 20, 20), 1)
            frame = cv2.circle(frame, (117 + self.offsetX, 42 + self.offsetY), 1, (20 if results[2] else 220, 220 if results[2] else 20, 20), 1)
            frame = cv2.circle(frame, (207 + self.offsetX, 42 + self.offsetY), 1, (20 if results[3] else 220, 220 if results[3] else 20, 20), 1)
            # # frame = cv2.circle(frame, (rand2, 13), 1, (220 if results[4] else 20, 220 if results[4] else 20, 20), 1)
            # # frame = cv2.circle(frame, (rand3, 42), 1, (220 if results[5] else 20, 220 if results[5] else 20, 20), 1)
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

    def isColorInRangeOfEarlyV1(self, bgr):
        result = False
        if len(bgr) == 4:
            b,g,r,a = (bgr)
        else:
            b,g,r = (bgr)

        if(b >  self.inhibitColorBoundaryV1[0][2] and b <  self.inhibitColorBoundaryV1[1][2]):#bluecheck
            if(g >  self.inhibitColorBoundaryV1[0][1] and g <  self.inhibitColorBoundaryV1[1][1]):#greencheck
                if(r >  self.inhibitColorBoundaryV1[0][0] and r <  self.inhibitColorBoundaryV1[1][0]):#redcheck
                    return True

        return result

    def isCorrectionInRangeOfEarlyV1(self, detected, screened):
        result = False
        if(self.debug):
            print("isCorrectionInRangeOfEarlyV1 Detected: ", detected)
            print("isCorrectionInRangeOfEarlyV1 Screened:", screened)

        subResult = []
        for i in range(len(detected)):
            b,g,r,a = (screened[i])
            b2,g2,r2,a2 = (detected[i])
            if(b >=  max(0, b2 - self.waitBoundary[0]) and b <= min(255, b2 + self.waitBoundary[0])):#bluecheck
                if(g >=  max(0, g2 - self.waitBoundary[1]) and g <= min(255, g2 + self.waitBoundary[1])):#bluecheck
                    if(r >=  max(0, r2 - self.waitBoundary[2]) and r <= min(255, r2 + self.waitBoundary[2])):#bluecheck
                        subResult.append(True)

        return len(subResult) == 2