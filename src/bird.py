import pygame
from pygame.locals import *
from neuralNetwork import NeuralNetwork

class Bird:

	def __init__(self):

		#Set Bird
		self.birdWidth = 34
		self.birdHeight = 24
		self.birdX = 50
		
		#Initialization Variables for Jump (Standard Jump)
		'''self.birdY = 200
		self.gravityAcceleration = 0	#self.isJumping = 0
		self.jumpHeight = 20
		self.vy=0'''

		#Initialization Variables for Jump (Parabolic Jump)
		self.birdY = 200
		self.gravityAcceleration = 1
		self.jumpHeight = -8
		self.vy=0

		#Initialization Variables for Life
		self.dead = 0

		#Initialization Variables for the neural Network
		self.birdPositionX = 0
		self.birdPositionY = 0
		self.normPosX = 0
		self.normPosY = 0
		self.distanceTraveled = 0

		#Initialization Variables for Score
		self.score = 0
		self.flagUpdateScore = 0

		#Initialization for Features
		self.iD = 0

		#Initialization number of jump made
		self.jumpMade=0

		#Initialization NN
		self.nn = NeuralNetwork()

		#Initialization GUI Bird
		self.bird = pygame.Rect(self.birdX, self.birdY, self.birdWidth, self.birdHeight)

		#Set GUI Bird
		self.birdImgs = [pygame.image.load("img/blueBirdFlap0.png").convert_alpha()]



	#uptdate the position of the bird(Standard Jump)
	'''def updateBird(self): #gli passo anche d=distanceBetweenPipes s=spaceBetweenPipes

		if self.nn.takeDecision(self.normPosX, self.normPosY) > 0.5:
			#if self.isJumping == 1:
			self.birdY -= self.jumpHeight
			self.gravityAcceleration = 0
			self.jumpMade+=1
		else:
			self.gravityAcceleration += 0.12
			self.birdY += self.gravityAcceleration

		self.bird = pygame.Rect(self.birdX, self.birdY, self.birdWidth, self.birdHeight)'''

#uptdate the position of the bird (Parabolic Jump)
	def updateBird(self,spaceBP):

		if self.nn.takeDecision(self.birdPositionX, self.birdPositionY,spaceBP) > 0.5:
			self.vy=self.jumpHeight
			self.jumpMade+=1


		self.vy+=self.gravityAcceleration
		self.birdY +=self.vy
		self.bird = pygame.Rect(self.birdX, self.birdY, self.birdWidth, self.birdHeight)