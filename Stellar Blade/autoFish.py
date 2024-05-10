import os
import cv2
import numpy as np
import math
from gtuner import *
import gtuner
import time
from random import randrange
import queue
import threading
from enum import Enum

'''
<version>1.0</version>

<shortdesc>
Stellar Blade: Auto Fish
Computer Vision : Detects button prompt for fishing
<i>Tested @ 1920x1080 input using PS Remote Play</i>
</shortdesc>

<keywords>Djmax Respect, Djmax, Computervision, CV, Auto, </keywords>

<donate>N/A</donate>
<docurl>N/A</docurl>
'''

class States(Enum):
	NOTFISHING = 'NotFishing'
	CASTING = 'Casting'
	BITEWAITING = 'BiteWaiting'
	HOOKING = 'Hooking'
	FISHING = 'Fishing'
	REELING = 'Reeling'
	CATCHING = 'Catching'
	RESETTING = 'Resetting'

class GCVWorker:
	def __init__(self, width, height):
		self.gcvdata = bytearray([0xFF]*6)
		# cast down and up left stick (3 sec)
		# Hook R2
		# Right stick (1 = left, 2 = right)
		# angle (unsigned byte from 1 - 255)
		# Lift up triangle (7 sec)
		# confirm X
		self.isDebug = True
		self.width = width
		self.height = height
		self.state = States.NOTFISHING
		self.gray = 0
		self.frame = 0
		self.old_gray = 0
		self.p0 = 0
		self.time = 0
		self.fps_counter = FPSCounter()

		# optical flow
		self.color = np.random.randint(0, 255, (100, 3))
		self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		# ref images
		self.refCastingGlyph = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/stellar_blade/templates/castingGlyph_canny.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.refCastingGlyph = cv2.Canny(self.refCastingGlyph,100,200)
		self.refCastingGlyph_height, self.refCastingGlyph_width = self.refCastingGlyph.shape[:2]
		self.refCastingGlyph_counter = 0

		self.refHookingGlyph = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/stellar_blade/templates/hookingGlyph_canny.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.refHookingGlyph = cv2.Canny(self.refHookingGlyph,100,200)
		self.refHookingGlyph_height, self.refHookingGlyph_width = self.refHookingGlyph.shape[:2]
		self.refHookingGlyph_counter = 0

		self.refReelInGlyph = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/stellar_blade/templates/reelinv2.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.refReelInGlyph = cv2.cvtColor(self.refReelInGlyph, cv2.COLOR_BGR2GRAY)
		self.refReelInGlyph_height, self.refReelInGlyph_width = self.refReelInGlyph.shape[:2]
		self.refReelInGlyph_counter = 0

		self.refLiftGlyph = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/stellar_blade/templates/liftGlyphv3.jpg', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.refLiftGlyph = cv2.cvtColor(self.refLiftGlyph, cv2.COLOR_BGR2GRAY)
		self.refLiftGlyph_height, self.refLiftGlyph_width = self.refLiftGlyph.shape[:2]
		self.refLiftGlyph_counter = 0

		self.prev_direction = "Left"
		self.left_counter = 0
		self.right_counter = 0
		self.square_counter = 0

		self.prev_angle = 0
		self.return_angle = 0
		self.prev_angle_counter = 0

	def process(self, frame):
		self.fps_counter.update()
		self.frame = frame
		self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		handleValue = self.handleState(self.state)
		self.checkState(self.state, frame)
		cv2.putText(self.frame, self.state.value, (3, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)
		cv2.putText(self.frame, "{:.2f}".format(self.fps_counter.get_fps()) + " fps", (557, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)
		if self.state == States.FISHING:
			cv2.putText(self.frame, self.prev_direction, (3, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)
		if self.state == States.REELING:
			cv2.putText(self.frame, "{:.2f}".format(self.return_angle) + " degrees", (3, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2, cv2.LINE_AA)
		return (self.frame, handleValue)

	# canny both then match template with consecutive frames needed to trigger
	def detectCastingGlyph(self, gray):
		threshold = 0.6
		consecutive_frames = 6
		didDraw = False

		edges = cv2.Canny(gray,100,200)
		result = cv2.matchTemplate(edges, self.refCastingGlyph, cv2.TM_CCOEFF_NORMED)

		loc = np.where(result >= threshold)
		for pt in zip(*loc[::-1]):
			if (self.isDebug):
				self.frame = cv2.rectangle(self.frame, pt, (pt[0] + self.refCastingGlyph_width, pt[1] + self.refCastingGlyph_height), (255, 255, 255), 2)
			didDraw = True
			break

		if (didDraw):
			self.refCastingGlyph_counter = self.refCastingGlyph_counter + 1
		else:
			self.refCastingGlyph_counter = 0

		if (self.refCastingGlyph_counter >= consecutive_frames):
			self.refCastingGlyph_counter = 0
			return True
		else:
			return False

	# canny both then match template with consecutive frames needed to trigger
	def detectHookingGlyph(self, gray):
		threshold = 0.6
		consecutive_frames = 6
		didDraw = False
		edges = cv2.Canny(gray,100,200)
		result = cv2.matchTemplate(edges, self.refHookingGlyph, cv2.TM_CCOEFF_NORMED)

		loc = np.where(result >= threshold)
		for pt in zip(*loc[::-1]):
			if (self.isDebug):
				self.frame = cv2.rectangle(self.frame, pt, (pt[0] + self.refHookingGlyph_width, pt[1] + self.refHookingGlyph_height), (255, 255, 255), 2)
			didDraw = True
			break

		if (didDraw):
			self.refHookingGlyph_counter = self.refHookingGlyph_counter + 1
		else:
			self.refHookingGlyph_counter = 0

		if (self.refHookingGlyph_counter >= consecutive_frames):
			self.refHookingGlyph_counter = 0
			return True
		else:
			return False

	# detect with match template only with consecutive frames needed to trigger
	def detectReelIn(self, gray):
		threshold = 0.8
		didDraw = False

		result = cv2.matchTemplate(self.gray, self.refReelInGlyph, cv2.TM_CCOEFF_NORMED)
		loc = np.where(result >= threshold)

		for pt in zip(*loc[::-1]):
			if (self.isDebug):
				self.frame = cv2.rectangle(self.frame, pt, (pt[0] + self.refReelInGlyph_width, pt[1] + self.refReelInGlyph_height), (255, 255, 255), 2)
			didDraw = True
			break

		return didDraw

	# detect direction with optical flow with a grid of points
	def detectMovementDirection(self, old_gray, frame_gray):
		consecutive_frames = 3
		self.p0 = self.create_uniform_points(old_gray, (20, 20))
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **self.lk_params)
		left_count = 0
		right_count = 0

		good_new = p1[st==1]
		good_old = self.p0[st==1]

		# Draw the tracks
		dx = 0
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			dx += a - c

			if(dx < 0):
				left_count += 1
			elif (dx > 0):
				right_count += 1

			# if (self.isDebug):
			# 	self.frame = cv2.line(self.frame, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2)
			# 	self.frame = cv2.circle(self.frame, (int(a), int(b)), 5, (255, 255, 255), -1)

		# Determine X direction
		if left_count > right_count:
			direction = "Left"
		elif right_count > left_count:
			direction = "Right"
		else:
			direction = "None"

		if (self.isDebug):
			total_count = left_count + right_count
			center_x = 386
			center_y = 133
			self.frame = cv2.arrowedLine(self.frame, (center_x, center_y), (center_x - round(left_count / total_count * (center_x - 20)), center_y), (255, 255, 0), 8)
			self.frame = cv2.arrowedLine(self.frame, (center_x, center_y), (center_x + round(right_count / total_count * (center_x - 20)), center_y), (0, 0, 255), 8)

		if(direction == "Left"):
			self.left_counter += 1
			self.right_counter = 0
		elif(direction == "Right"):
			self.right_counter += 1
			self.left_counter = 0

		if (self.left_counter >= 3):
			self.prev_direction = "Left"
		if (self.right_counter >= 3):
			self.prev_direction = "Right"

		self.p0 = good_new.reshape(-1, 1, 2)

	def detectSquare(self, gray):
		consecutive_frames = 4
		didDraw = False

		edges = cv2.Canny(gray,100,200)
		contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			epsilon = 0.01 * cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, epsilon, True)

			if len(approx) == 4:
				(x, y, w, h) = cv2.boundingRect(approx)
				ar = w / float(h)
				if 0.95 < ar < 1.05 and w > 25 and w < 39:  # Aspect ratio for squareness
					if (self.isDebug):
						self.frame = cv2.drawContours(self.frame, [approx], 0, (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 3)
					didDraw = True
					break

		if (didDraw):
			self.square_counter = self.square_counter + 1
		else:
			self.square_counter = 0

		if (self.square_counter >= consecutive_frames):
			self.square_counter = 0
			return True
		else:
			return False

	def detectLiftGlyph(self, gray):
		threshold = 0.8
		didDraw = False

		result = cv2.matchTemplate(self.gray, self.refLiftGlyph, cv2.TM_CCOEFF_NORMED)
		loc = np.where(result >= threshold)

		for pt in zip(*loc[::-1]):
			if (self.isDebug):
				self.frame = cv2.rectangle(self.frame, pt, (pt[0] + self.refLiftGlyph_width, pt[1] + self.refLiftGlyph_height), (0, 0, 0), 2)
			didDraw = True
			break

		return didDraw

	def detectReelInAngle(self, gray, frame):
		consecutive_frames = 3
		didDraw = False

		circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=200, param2=25, minRadius=30, maxRadius=80)
		result = frame
		center = (0,0)
		houghCircles_count = 0
		mask = np.zeros_like(self.gray)

		if circles is not None:
			circles = np.round(circles[0, :]).astype("int")
			houghCircles_count = len(circles)
			for (x, y, r) in circles:
				center = (x,y)
				cv2.circle(mask, (x, y), r + 10, (255, 255, 255), -1)

			# Apply the mask to the original image
			result = cv2.bitwise_and(frame, frame, mask=mask)
		# print(center, houghCircles_count)
		if (center != (0,0) and houghCircles_count == 1):
			lower_color = np.array([90, 44, 166])
			upper_color = np.array([100, 196, 255])

			# Create a mask based on the color threshold
			hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_color, upper_color)

			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			if contours:
				didDraw = True
				angle = 0
				minDeg = 360
				maxDeg = 0
				for contour in contours:
					for ptsList in contour:
						pts = ptsList[0]
						# Calculate the vector from the center to the first point
						vector_x = pts[0] - center[0]
						vector_y = pts[1] - center[1]

						# Calculate the angle of the vector relative to the positive x-axis
						angle_degrees = np.degrees(np.arctan2(vector_y, vector_x)) + 90

						# Ensure angle is positive (range: 0 to 360 degrees)
						degrees = (angle_degrees + 360) % 360
						if (degrees <= minDeg):
							minDeg = degrees
						if (degrees >= maxDeg):
							maxDeg = degrees

				angle = (minDeg + maxDeg) / 2

			if(didDraw and self.prev_angle == 0 and self.return_angle == 0):
				self.prev_angle = angle
				self.return_angle = angle

			if (didDraw):
				if(self.inRange(self.prev_angle, angle, 5) or self.prev_angle_counter >= consecutive_frames):
					self.prev_angle = angle
					self.return_angle = angle
				else:
					self.prev_angle = angle
					if(self.inRange(self.prev_angle, angle, 5)):
						self.prev_angle_counter = 1
					else:
						self.prev_angle_counter = 0

		if (self.isDebug):
			center_x = 368
			center_y = 460
			radius = 50
			endpoint_x = center_x + int(radius * math.sin(math.radians(self.return_angle)))
			endpoint_y = center_y - int(radius * math.cos(math.radians(self.return_angle)))
			self.frame = cv2.arrowedLine(self.frame, (center_x, center_y), (endpoint_x, endpoint_y), (255, 0, 255), 5)

	def inRange(self, old, new, thresh):
		diff = np.abs(old - new)
		return diff <= thresh

	def create_uniform_points(self, image, grid_size):
		points = []
		step_y, step_x = image.shape[0] // grid_size[1], image.shape[1] // grid_size[0]
		y, x = step_y // 2, step_x // 2
		while y < image.shape[0]:
			x = step_x // 2
			while x < image.shape[1]:
				points.append((x, y))
				x += step_x
			y += step_y
		return np.array(points, np.float32).reshape(-1, 1, 2)

	def handleState(self, state):
		if state == States.NOTFISHING:
			# DO SOMETHING?
			return None
		elif state == States.CASTING:
			print("changing from ", self.state.value, " to ", States.BITEWAITING.value)
			self.state = States.BITEWAITING
			self.time = time.time()
			# SEND Casting Input (LS down then up)
			return bytearray([0x01, 0x00, 0x00, 0x00, 0x00, 0x00])
		elif state == States.BITEWAITING:
			# DO SOMETHING?
			return None
		elif state == States.HOOKING:
			print("changing from ", self.state.value, " to ", States.FISHING.value)
			self.state = States.FISHING
			# SEND Casting Input (R2)
			return bytearray([0x00, 0x01, 0x00, 0x00, 0x00, 0x00])
		elif state == States.FISHING:
			# Send counter input based on self.prev_direction
			if (self.prev_direction == "Left"):
				return bytearray([0x00, 0x00, 0x01, 0x00, 0x00, 0x00])
			elif (self.prev_direction == "Right"):
				return bytearray([0x00, 0x00, 0x02, 0x00, 0x00, 0x00])
			return bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
		elif state == States.REELING:
			# TODO: send variable R2
			intVal = int(np.ceil(self.return_angle * 255 / 360))
			byteVal = intVal.to_bytes(1, 'big', signed=False)
			return bytearray([0x00, 0x00, 0x00, byteVal[0], 0x00, 0x00])
		elif state == States.CATCHING:
			self.time = time.time()
			print("changing from ", self.state.value, " to ", States.RESETTING.value)
			self.state = States.RESETTING
			# send triangle input
			return bytearray([0x00, 0x00, 0x00, 0x00, 0x01, 0x00])
		elif state == States.RESETTING:
			if (self.time + 9 < time.time()):
				print("changing from ", self.state.value, " to ", States.NOTFISHING.value)
				self.state = States.NOTFISHING
				# send cross input
				return bytearray([0x00, 0x00, 0x00, 0x00, 0x00, 0x01])
			return None

	def checkState(self, state, frame):
		if state == States.NOTFISHING:
			result = self.detectCastingGlyph(self.gray)
			if (result):
				print("changing from ", self.state.value, " to ", States.CASTING.value)
				self.state = States.CASTING
			return ""
		elif state == States.CASTING:
			# CHECK SOMETHING
			return ""
		elif state == States.BITEWAITING:
			if (self.time + 3 < time.time()):
				result = self.detectHookingGlyph(self.gray)
				if (result):
					print("changing from ", self.state.value, " to ", States.HOOKING.value)
					self.state = States.HOOKING
			return ""
		elif state == States.HOOKING:
			# CHECK SOMETHING
			return ""
		elif state == States.FISHING:
			result = self.detectReelIn(self.gray)
			if (result):
				print("changing from ", self.state.value, " to ", States.REELING.value)
				self.state = States.REELING
			if (type(self.old_gray).__name__ != "int"):
				self.detectMovementDirection(self.gray, self.old_gray)
			self.old_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
			return ""
		elif state == States.REELING:
			result = self.detectSquare(self.gray)
			if (result):
				print("changing from ", self.state.value, " to ", States.FISHING.value)
				self.state = States.FISHING
			result = self.detectLiftGlyph(self.gray)
			if (result):
				print("changing from ", self.state.value, " to ", States.CATCHING.value)
				self.state = States.CATCHING
			self.detectReelInAngle(self.gray, frame)
			return ""
		elif state == States.CATCHING:
			# CHECK SOMETHING
			return ""
		elif state == States.RESETTING:
			# CHECK SOMETHING
			return ""

	def __del__(self):
		del self.gcvdata
		del self.isDebug
		del self.width
		del self.height
		del self.state
		del self.gray
		del self.old_gray
		del self.p0
		del self.time
		del self.fps_counter
		del self.color
		del self.lk_params
		del self.refCastingGlyph
		del self.refCastingGlyph_height
		del self.refCastingGlyph_counter
		del self.refHookingGlyph
		del self.refHookingGlyph_height
		del self.refHookingGlyph_counter
		del self.refReelInGlyph
		del self.refReelInGlyph_height
		del self.refReelInGlyph_counter
		del self.refLiftGlyph
		del self.refLiftGlyph_height
		del self.refLiftGlyph_counter
		del self.prev_direction
		del self.left_counter
		del self.right_counter
		del self.square_counter
		del self.prev_angle
		del self.return_angle
		del self.prev_angle_counter

class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:  # Update FPS every 1 second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def get_fps(self):
        return self.fps