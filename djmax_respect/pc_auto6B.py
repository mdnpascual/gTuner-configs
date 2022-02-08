import os
import cv2
import numpy as np
from gtuner import *
import gtuner
import time
from random import randrange
import queue
import threading
import time

class GCVWorker:
    def __init__(self, width, height):
        self.gcvdata = bytearray([0xFF]*8)
        self.keyTiming = [0,0,0,0,0,0,0,0]
        self.width = width
        self.height = height
        # Use this if imread doesn't work: cv2.imdecode(np.fromfile('<fullpath>', dtype=np.uint8), -1)
        self.left8b = cv2.imread('template/left8b.jpg')
        self.right8b = cv2.imread('template/right8b.jpg')
        self.white6b = cv2.imread('template/whiteNote.jpg')
        self.color6b = cv2.imread('template/colorNote.jpg')
        self.whiteHoldS6b = cv2.imread('template/whiteHoldS.jpg')
        self.whiteHoldM6b = cv2.imread('template/whiteHoldM.jpg')
        self.whiteHoldE6b = cv2.imread('template/whiteHoldE.jpg')
        self.colorHoldS6b = cv2.imread('template/colorHoldS.jpg')
        self.colorHoldM6b = cv2.imread('template/colorHoldM.jpg')
        self.colorHoldE6b = cv2.imread('template/colorHoldE.jpg')
        self.threshold = 0.85
        self.isDebug = False

    def process(self, frame):
        clean = frame.copy()
        q = queue.Queue()
        worker = [None] * 8
        results = [0.00] * 8

        for i in range(6):
            if(i == 0):
                worker[i] = threading.Thread(target=self.detectFirstWhite6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 1):
                worker[i] = threading.Thread(target=self.detectSecondWhite6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 2):
                worker[i] = threading.Thread(target=self.detectThirdWhite6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 3):
                worker[i] = threading.Thread(target=self.detectFourthWhite6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 4):
                worker[i] = threading.Thread(target=self.detectFirstColor6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 5):
                worker[i] = threading.Thread(target=self.detectSecondColor6b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 6):
                worker[i] = threading.Thread(target=self.detectLeft8b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 7):
                worker[i] = threading.Thread(target=self.detectRight8b, args=(q, frame, clean, results, i,), daemon=True)
            worker[i].start()
            q.put(i)

        if not self.isDebug:
            q.join()
        return (frame, self.processData(results)) if not self.isDebug else (frame, None)

    def processData(self, resultArr):
        buildArray = [0x00]*len(resultArr)
        for i in range(len(resultArr)):
            buildArray[i] = 0x01 if resultArr[i] >= self.threshold else 0x00

        return bytearray(buildArray)

    def genericDetect(self, frame, clean, template, xStart, xEnd, debugName, debugX, debugY, activeColor, activeOnly, q):
        t0 = time.perf_counter()
        cleanCropped = clean[0:self.height, xStart:xEnd]
        chan,w,h = template.shape[::-1]
        result = cv2.matchTemplate(cleanCropped, template, 5)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        if(self.threshold < max_val):
            frame = cv2.rectangle(frame,(top_left[0] + xStart, top_left[1]), (bottom_right[0] + xStart, bottom_right[1]), 255, 2)
        t1 = time.perf_counter()
        if(self.threshold < max_val or activeOnly):
            cv2.putText(frame, debugName + str('{0:.2f}'.format(max_val)) + str(top_left) + " " + '{0:.5f}'.format(1000*(t1-t0)) + "ms", (debugX, debugY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, activeColor if self.threshold < max_val else (255, 255, 255), 1, cv2.LINE_AA)
        return max_val if not self.isDebug else max_val
        # return cleanCropped if self.isDebug else frame

    def detectFirstWhite6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS6b, 20, 183, "6B1WHS: ", 2, 13, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM6b, 20, 183, "6B1WHM: ", 2, 13, 	(0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE6b, 20, 183, "6B1WHE: ", 2, 13, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white6b, 20, 183, "6B1W: ", 2, 13, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()
        # self.genericDetect(frame, clean, self.whiteHoldS6b, 20, 183, "6B1HS: ", 2, 13, (0, 255, 0), True, q)
        # self.genericDetect(frame, clean, self.whiteHoldM6b, 20, 183, "6B1HM: ", 2, 33, 	(0, 215, 255), True, q)
        # self.genericDetect(frame, clean, self.whiteHoldE6b, 20, 183, "6B1HE: ", 2, 53, (0, 255, 255), True, q)
        # self.genericDetect(frame, clean, self.white6b, 20, 183, "6B1W: ", 2, 73, (0, 255, 0), True, q)

    def detectSecondWhite6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS6b, 340, 503, "6B2WHS: ", 2, 53, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM6b, 340, 503, "6B2WHM: ", 2, 53, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE6b, 340, 503, "6B2WHE: ", 2, 53, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white6b, 340, 503, "6B2W: ", 2, 53, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.white6b, 338, 501, "6B2W: ", 2, 53, q)

    def detectThirdWhite6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS6b, 503, 666, "6B3WHS: ", 2, 73, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM6b, 503, 666, "6B3WHM: ", 2, 73, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE6b, 503, 666, "6B3WHE: ", 2, 73, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white6b, 503, 666, "6B3W: ", 2, 73, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.white6b, 503, 666, "6B3W: ", 2, 73, q)

    def detectFourthWhite6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS6b, 822, 985, "6B4WHS: ", 2, 113, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM6b, 822, 985, "6B4WHM: ", 2, 113, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE6b, 822, 985, "6B4WHE: ", 2, 113, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white6b, 822, 985, "6B4W: ", 2, 113, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.white6b, 822, 985, "6B4W: ", 2, 113, q)

    def detectFirstColor6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.colorHoldS6b, 181, 344, "6B1CHS: ", 2, 33, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.colorHoldM6b, 181, 344, "6B1CHM: ", 2, 33, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.colorHoldE6b, 181, 344, "6B1CHE: ", 2, 33, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.color6b, 181, 344, "6B1C: ", 2, 33, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.color6b, 181, 344, "6B1C: ", 2, 33, q)

    def detectSecondColor6b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.colorHoldS6b, 662, 825, "6B2CHS: ", 2, 93, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.colorHoldM6b, 662, 825, "6B2CHM: ", 2, 93, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.colorHoldE6b, 662, 825, "6B2CHE: ", 2, 93, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.color6b, 662, 825, "6B2C: ", 2, 93, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.color6b, 662, 825, "6B2C: ", 2, 93, q)

    def detectRight8b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        results[i] = self.genericDetect(frame, clean, self.right8b, 500, 1000, "8BR: ", 2, 133, (0, 255, 0), True, q)
        if not self.isDebug:
            q.task_done()

    def detectLeft8b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        results[i] = self.genericDetect(frame, clean, self.left8b, 19, 505, "8BL: ", 2, 153, (0, 255, 0), True, q)
        if not self.isDebug:
            q.task_done()

    def samplePerformance(self, frame):
        t0 = time.perf_counter()
        frame = frame[0:self.height, 500:1000]
        t1 = time.perf_counter()
        print("Function=%s, Time=%s" % ('crop', t1 - t0))
        print("Function=%s, Time=%s" % ('Total', t1 - t0))

    def __del__(self):
        del self.gcvdata
        del self.keyTiming
        del self.width
        del self.height
        del self.left8b
        del self.right8b
        del self.white6b
        del self.color6b
        del self.whiteHoldS6b
        del self.whiteHoldM6b
        del self.whiteHoldE6b
        del self.colorHoldS6b
        del self.colorHoldM6b
        del self.colorHoldE6b
        del self.threshold
        del self.isDebug
