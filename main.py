import cv2
import random
import numpy as np
import mediapipe as mp
# from constants import *
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 500

GRAVITY = 1
class Ball:
	'''
	This class is used to create a ball, the 
	position of the ball is updated in this 
	class.
	'''
	def __init__(self):
		'''
		This is the initializer definition,
		when a new ball is created this function 
		is called. This functions assigns the ball
		a Y value of 0 a radius value of 15 and an
		X value that is randomly chosen between 0 
		and the screen width.
		'''
		self.x = random.randint(0, SCREEN_WIDTH)
		self.y = 0
		self.r = 15
		self.vel_x = 0
		self.vel_y = 0
		self.color = (255, 0, 0)

	def set_vel(self, vel_x, vel_y):
		'''
		This function sets the ball velocity
		'''
		self.vel_y = vel_y
		self.vel_x = vel_x

	def get_color(self):
		'''
		The ball color is extracted using this funciton
		'''
		return self.color

	def get_position(self):
		'''
		This function returns the x and y position
		of the ball
		'''
		return self.x, self.y

	def set_position(self, x=None, y=None):
		'''
		The x and y position can be set by the user
		by calling this funciton
		'''
		if x != None:
			self.x = x
		
		if y != None:
			self.y = y

	def get_radius(self):
		'''
		This function returns the radius of the ball
		'''
		return self.r

	def update_position(self, acc_x, acc_y):
		'''
		This function updates the position of the ball.
		The ball x and y position are updated by adding
		the x and y velocities to the ball's current x 
		and y position. Some additional forces also act 
		on the ball, one of these forces is the Gravity.
		The gravity is applied to the ball every time
		this funciton is called and the ball's y position
		changes accordingly.
		Additionally, acceleration in the x or y direction
		may be applied to the ball, the y acceleration can
		be caused by the ball hitting the paddle, which 
		causes the ball the accelerate in the upward direction.
		The x acceleration can be caused by the movement of
		the paddle that is transfered to the ball.
		'''
		self.x += self.vel_x
		self.y += self.vel_y

		self.vel_x += acc_x
		self.vel_y += (GRAVITY + acc_y)

class Paddle:
	'''
	This is similar to the ball class except this
	class represents the paddle that the player controls
	to interact with the balls.
	'''
	def __init__(self):
		'''
		The paddle's position is set to the bottom middle of the 
		screen.
		'''
		self.w = 100
		self.h = 30
		self.x = SCREEN_WIDTH//2
		self.y = SCREEN_HEIGHT - self.h//2
		self.prev_x = self.x
		self.vel_x = 0

	def get_position(self):
		'''
		Returns the position of the paddle
		'''
		return self.x, self.y

	def get_size(self):
		# Returns the paddle size (width and height)
		return self.w, self.h

	def update_position(self, position_norm):
		# Given the position of the hand, update the 
		# position of the paddle. This function also
		# updates the x velocity of paddle.
		self.prev_x = self.x
		if position_norm is not None:
			position_x = int(position_norm*SCREEN_WIDTH)
			self.x = position_x
		self.vel_x = self.x - self.prev_x

	def collision_det(self, ball):
		'''
		Given a ball, this function determines whether the ball is 
		hitting the paddle or not. If the ball is hitting the paddle
		then the ball is bounced upwards and if the paddle has any
		x velocity it is transferred to the ball.
		'''
		acc_x = 0
		acc_y = 0
		x, y = ball.get_position()
		r = ball.get_radius()

		if y+r>=self.y and y-r<=self.y:
			if (x+r)>self.x and (x-r)<(self.x+self.w):
				acc_y = -1*abs(int(0.5*ball.vel_y))
				if ball.vel_y>0:
					ball.update_position(acc_x, acc_y)
					return True
					# ball.set_vel(0, 0)
					# ball.set_position(y = self.y - r)
					# acc_x = self.vel_x
		ball.update_position(acc_x, acc_y)
		return False


class Draw:
	'''
	This class takes care of all the rendering. The player and the paddle are
	rendered by this class.
	'''
	def __init__(self):
		'''
		When this class is initialized a canvas is created on which the objects
		will be drawn
		'''
		self.canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3)).astype("uint8")

	def clear(self):
		'''
		This function clears the canvas
		'''
		self.canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3)).astype("uint8")

	def draw_entities(self, entities):
		'''
		This function goes over all the balls and extracts that ball's x, y and radius
		and draws that ball onto the canvas.
		'''
		for entity in entities:
			x, y = entity.get_position()
			r = entity.get_radius()
			color = entity.get_color()
			cv2.circle(self.canvas, (x, y), r, color, -1)
			cv2.circle(self.canvas, (x, y), r, (0,0,0), 2)

	def draw_player(self, player):
		'''
		This function takes the player and gets its dimentions and
		postion and draws it on the canvas,
		'''
		x, y = player.get_position()
		w, h = player.get_size()
		cv2.rectangle(self.canvas, (x, y), (x+w, y+h), (0, 255, 0), -1)

	def overlay_hand(self, frame):
		h, w, _ = frame.shape
		start_x = SCREEN_WIDTH - w
		start_y = SCREEN_HEIGHT - (h + 100)
		self.canvas[start_y:start_y+h, start_x:start_x+w,:] = frame

	def display_frame(self):
		'''
		This function finally shows the canvas where everything has been drawn.
		'''
		cv2.imshow("Canvas", self.canvas)

class HandDetector:
	'''
	This class extracts a frame from the camera and then uses media pipe 
	to get the position of the hand in the frame.
	'''
	def __init__(self):
		'''
		When the hand detector class is initilzed this function opens the webcam
		and sets up the hand detector.
		'''
		self.cap = cv2.VideoCapture(0)
		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands()
		self.mpDraw = mp.solutions.drawing_utils

	def get_hand_position(self):
		'''
		This function extracts an image from the webcam and then passes it through
		the mediapipe module. The module detects the hand position and then normalizes
		the position of the hand in the camera frame which is used to move the paddle.
		'''
		ret, frame = self.cap.read()
		frame = cv2.flip(frame, 1)
		imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.hands.process(imgRGB)
		position_norm = None
		if results.multi_hand_landmarks:
			for handLms in results.multi_hand_landmarks:
				min_x = 10000
				min_y = 10000
				max_x = 0
				max_y = 0
				for id, lm in enumerate(handLms.landmark):
					h, w, c = frame.shape
					cx, cy = int(lm.x *w), int(lm.y*h)

					if cx>max_x:
						max_x = cx
					if cx<min_x:
						min_x = cx

					if cy>max_y:
						max_y = cy
					if cy<min_y:
						min_y = cy


				mid_x = min_x + (max_x - min_x)//2
				mid_y = min_y + (max_y - min_y)//2
				position_norm = mid_x/w
				cv2.circle(frame, (mid_x,mid_y), 7, (255,0,255), cv2.FILLED)
				cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
		# cv2.imshow("Image", frame)
		h, w, _ = frame.shape
		frame = cv2.resize(frame, (int(w*0.3), int(h*0.3)))
		return position_norm, frame

class Game:
	'''
	This class contains the game logic.
	'''
	def __init__(self):
		'''
		When initialzed the draw class, the player class and the hand detector class
		is initialized.
		'''
		self.canvas = Draw()
		self.player = Paddle()
		self.hand_detector = HandDetector()
		self.balls = []

	def add_ball(self):
		'''
		This function adds a ball to the game.
		'''
		self.balls.append(Ball())

	def step(self, position_norm):
		'''
		This function steps the game. It updates the ball positions and detects
		collisions with the player. This function also updates the paddle position
		based on the position of the hand.
		'''
		self.player.update_position(position_norm)
		
		for ball in self.balls:
			if (ball.y + ball.r)>SCREEN_HEIGHT:
				self.balls.remove(ball)
				continue
			collision = self.player.collision_det(ball)
			if collision:
				self.balls.remove(ball)

	def mainloop(self):
		'''
		This is the main loop of the game. This function first clears the canvas, then draws 
		the balls on it and then draws the paddle on the canvas. Then the position of the hand
		is extracted and it is used to move the paddle.
		'''
		while True:
			if random.random()>0.9:
				self.add_ball()
			self.canvas.clear()
			self.canvas.draw_entities(self.balls)
			self.canvas.draw_player(self.player)
			position_norm, frame = self.hand_detector.get_hand_position()
			self.canvas.overlay_hand(frame)
			self.canvas.display_frame()
			self.step(position_norm)
			k = cv2.waitKey(30)
			if k == ord("q"):
				break

game = Game()
game.add_ball()
game.mainloop()
