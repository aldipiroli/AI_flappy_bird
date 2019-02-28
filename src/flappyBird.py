from __future__ import division

import pygame
from pygame.locals import *
import sys
import random
import time
from bird import Bird
from geneticAlgorithm import GeneticAlgorithm
from learningAlgorithm import LearningAlgorithm


class FlappyBird:

	def __init__(self):
		# Control Writing
		self.doIWrtie = 0

		#Set Screen	
		self.WIDTH = 288 * 2
		self.HEIGHT = 512
		self.SCORES = 288 			#Dim Score Window
		self.interlineSCORES = 30 	#Interline between Scores
		self.baseWidth = 336		#Width Ground		
		self.baseHeight = 112		#Height Ground

		#Set Time
		self.clockSpeed = 250
		
		#Set Pipes
		self.pipeWidth = 52		
		self.pipeHeight = 320
		self.pipeSpeed = 5
		self.spaceBetweenPipesMin = 150
		self.spaceBetweenPipesMax = 200
		self.distanceBetweenPipesMin=self.WIDTH/2
		self.distanceBetweenPipesMax=(3/4)*self.WIDTH
		self.distanceBetweenPipes=0
		self.DistanceBetweenPipesValue =self.WIDTH / 2 - self.pipeWidth / 2 

		'''self.spaceBetweenPipes0 = random.randint(self.spaceBetweenPipesMin, self.spaceBetweenPipesMax)
		self.spaceBetweenPipesHeight0 = random.randint(self.pipeHeight - self.spaceBetweenPipesMax / 2, self.pipeHeight + self.spaceBetweenPipesMin / 2)'''
		self.spaceBetweenPipes0=250
		self.spaceBetweenPipesHeight0=300
		self.spaceBetweenPipes1 = random.randint(int(self.spaceBetweenPipesMin), int(self.spaceBetweenPipesMax))
		self.spaceBetweenPipesHeight1 = random.randint(int(self.pipeHeight - self.spaceBetweenPipesMax / 2), int(self.pipeHeight + self.spaceBetweenPipesMin / 2))

		#Initialization Variables for Pipes
		self.pipeX0 = self.WIDTH / 2 - self.pipeWidth / 2 
		self.pipeX1 = self.WIDTH
		self.pipeClosest = self.pipeX0
		self.nextSpaceBetweenPipes=self.spaceBetweenPipes0

		#Initialization Variables for Scores
		self.flagPipePassed = 0
		self.flagScoresIncremented = 0
		self.maxScore=0
		self.maxFinalScore=400
 
		#Set GUI Screen
		self.screen = pygame.display.set_mode((self.WIDTH + self.SCORES, self.HEIGHT + self.baseHeight))
		self.background = pygame.image.load("img/backgroundDay.png").convert()
		self.base = pygame.image.load("img/base.png").convert()
		self.gameOver = pygame.image.load("img/gameover.png").convert()
		self.topPipe = pygame.image.load("img/greenPipeTop.png").convert_alpha()
		self.downPipe = pygame.image.load("img/greenPipeDown.png").convert_alpha()
		self.backgroundScores = pygame.image.load("img/whiteRect.png").convert()

		#Create Flock
		self.numberOfBirds=2
		self.Birds = []
		for i in range(1, self.numberOfBirds+1):
			self.Birds.append(Bird())

		#Indexing of birds
		_j = 1
		for bird in self.Birds:
			bird.iD = _j
			_j += 1

		#Vector of alive birds	
		self.birdsAlive = [bird for bird in self.Birds if bird.dead == 0]

		self.genetic=GeneticAlgorithm(self.Birds)
		self.geneticCycles=0
		'''self.NeuralNetworks = []
		for i in range(1, 11):
			self.NeuralNetworks.append(NeuralNetwork())'''

	# --- updatePipes --- 

	def updatePipes(self):
		self.pipeX0 -= self.pipeSpeed
		self.pipeX1 -= self.pipeSpeed

		if self.pipeX0 < - self.pipeWidth:
			#self.distanceBetweenPipes=450
			self.distanceBetweenPipes=random.randint(int(self.distanceBetweenPipesMin), int(self.distanceBetweenPipesMax))
			self.spaceBetweenPipesHeight0 = random.randint(int(self.pipeHeight - self.spaceBetweenPipesMax / 2), int(self.pipeHeight + self.spaceBetweenPipesMin / 2))
			self.spaceBetweenPipes0 = random.randint(int(self.spaceBetweenPipesMin), int(self.spaceBetweenPipesMax))
			self.pipeX0 = self.pipeX1 + self.distanceBetweenPipes
			self.flagPipePassed = 0
			for Bird in self.Birds:
				Bird.flagUpdateScore = 0
		if self.pipeX1 <- self.pipeWidth:
			#self.distanceBetweenPipes=450
			self.distanceBetweenPipes=random.randint(int(self.distanceBetweenPipesMin), int(self.distanceBetweenPipesMax))
			self.spaceBetweenPipesHeight1 = random.randint(int(self.pipeHeight - self.spaceBetweenPipesMax / 2), int(self.pipeHeight + self.spaceBetweenPipesMin / 2))
			self.spaceBetweenPipes1 = random.randint(int(self.spaceBetweenPipesMin), int(self.spaceBetweenPipesMax))
			self.pipeX1 = self.pipeX0 + self.distanceBetweenPipes
			self.flagPipePassed = 0
			for Bird in self.Birds:
				Bird.flagUpdateScore = 0

	def nextPipeX(self):
		if ((self.pipeX0 >= 67 -(self.pipeWidth/2)) and (self.pipeX0 < self.pipeX1 or self.pipeX1<=67-(self.pipeWidth/2))): #67=birdX+birdWidth/2
			return self.pipeX0
		else:
			return self.pipeX1

	def nextPipeY(self):
		if ((self.pipeX0 >= 67 -(self.pipeWidth/2)) and (self.pipeX0 < self.pipeX1 or self.pipeX1<=67-(self.pipeWidth/2))):
			return self.spaceBetweenPipesHeight0
		else:
			return self.spaceBetweenPipesHeight1

	def birdsPositions(self): #distance from the bird's center 
		for Bird in self.Birds:
			if Bird.dead == 0:
				Bird.birdPositionX = (self.nextPipeX() + self.pipeWidth/2) - (Bird.birdX + Bird.birdWidth/2)
				Bird.birdPositionY = self.nextPipeY() - (Bird.birdY	+ Bird.birdHeight/2)
			
	def getNextSpaceBetweenPipes(self):
		if ((self.pipeX0 >= 67 -(self.pipeWidth/2)) and (self.pipeX0 < self.pipeX1 or self.pipeX1<=67-(self.pipeWidth/2))):
			return self.spaceBetweenPipes0
		else:
			return self.spaceBetweenPipes1
		


	def getDistanceBetweenPipes(self):
		if (self.pipeX0 <= 67-self.pipeWidth/2):
			self.DistanceBetweenPipesValue=self.pipeX1-self.pipeX0
		if (self.pipeX1 <=67-self.pipeWidth/2):
			self.DistanceBetweenPipesValue=self.pipeX0-self.pipeX1
		return self.DistanceBetweenPipesValue

	# --- updateGame ---

	def updateLife(self):
		downPipeRect0 = pygame.Rect(self.pipeX0, self.spaceBetweenPipesHeight0 + self.spaceBetweenPipes0 / 2, self.pipeWidth, self.pipeHeight)
		topPipeRect0 = pygame.Rect(self.pipeX0, 0 - self.pipeHeight + self.spaceBetweenPipesHeight0 - self.spaceBetweenPipes0 / 2, self.pipeWidth, self.pipeHeight)
		downPipeRect1 = pygame.Rect(self.pipeX1, self.spaceBetweenPipesHeight1 + self.spaceBetweenPipes1 / 2, self.pipeWidth, self.pipeHeight)
		topPipeRect1 = pygame.Rect(self.pipeX1, 0 - self.pipeHeight + self.spaceBetweenPipesHeight1 - self.spaceBetweenPipes1 / 2, self.pipeWidth, self.pipeHeight)

		for Bird in self.Birds:
			if downPipeRect0.colliderect(Bird.bird) or topPipeRect0.colliderect(Bird.bird) or downPipeRect1.colliderect(Bird.bird) or topPipeRect1.colliderect(Bird.bird) or Bird.birdY > self.HEIGHT - Bird.birdHeight / 2 or Bird.birdY < Bird.birdHeight / 2:
				Bird.dead = 1

	def updateScores(self):

		for Bird in self.Birds:
			if Bird.birdX > self.pipeClosest + self.pipeWidth and Bird.flagUpdateScore == 0 and self.flagPipePassed == 0 and Bird.dead == 0:
				Bird.score += 1
				Bird.flagUpdateScore = 1
				self.flagScoresIncremented += 1
		if self.flagScoresIncremented != 0:
			self.flagPipePassed = 1
			self.flagScoresIncremented = 0

	def gui(self):
		#Font
		font = pygame.font.SysFont("Arial", 25)
		#GUI
		self.screen.fill((255, 255, 255))
		self.screen.blit(self.background, (0, 0))
		self.screen.blit(self.background, (self.WIDTH / 2, 0))
		#Pipe0 Image
		self.screen.blit(self.downPipe, (self.pipeX0, self.spaceBetweenPipesHeight0 + self.spaceBetweenPipes0 / 2))
		self.screen.blit(self.topPipe, (self.pipeX0, 0 - self.pipeHeight + self.spaceBetweenPipesHeight0 - self.spaceBetweenPipes0 / 2))
		#Pipe1 Image
		self.screen.blit(self.downPipe, (self.pipeX1, self.spaceBetweenPipesHeight1 + self.spaceBetweenPipes1 / 2))
		self.screen.blit(self.topPipe, (self.pipeX1, 0 - self.pipeHeight + self.spaceBetweenPipesHeight1 - self.spaceBetweenPipes1 / 2))
		#Base Image
		self.screen.blit(self.base, (0, self.HEIGHT))
		self.screen.blit(self.base, (self.WIDTH / 2, self.HEIGHT))
		#Score 
		scoreRect = pygame.Rect(self.WIDTH, 0, self.SCORES, self.HEIGHT + self.baseHeight)
		self.screen.blit(self.backgroundScores, (self.WIDTH, 0, self.SCORES, self.HEIGHT + self.baseHeight))
		self.interlineSCORES = 30
		#Score,Genetic Cycles and Birds Alive printing on screen
		self.screen.blit(font.render("Genetic Cycles" + " : " + str(self.geneticCycles), -1, (0, 0, 0)), (self.WIDTH + 30, self.interlineSCORES))
		self.interlineSCORES += 50
		for Bird in self.Birds:
			if(Bird.score>self.maxScore):
				self.maxScore=Bird.score
		self.screen.blit(font.render("Birds alive" + " : " + str(len(self.birdsAlive)) + "/ " + str(len(self.Birds)), -1, (0, 0, 0)), (self.WIDTH + 30, self.interlineSCORES))
		self.interlineSCORES += 50
		self.screen.blit(font.render("Max Score" + " : " + str(self.maxScore), -1, (0, 0, 0)), (self.WIDTH + 30, self.interlineSCORES))
		self.interlineSCORES += 50

	def difficultyIncrease(self):
		#decreasing space between pipes

		if self.maxScore>100 and (self.maxScore%50)==0 and self.maxScore!=0 and (self.pipeX0==self.WIDTH / 2 - self.pipeWidth or self.pipeX1==self.WIDTH / 2 - self.pipeWidth):
			if self.spaceBetweenPipesMax>self.spaceBetweenPipesMin+10 and self.spaceBetweenPipesMin>100 :
				self.spaceBetweenPipesMax-=10
				self.spaceBetweenPipesMin-=5

	# --- run ---
	def restart(self):
		self.spaceBetweenPipesMin = 150
		self.spaceBetweenPipesMax = 200
		self.distanceBetweenPipesMin=self.WIDTH/2
		self.distanceBetweenPipesMax=(3/4)*self.WIDTH

		'''self.spaceBetweenPipes0 = random.randint(self.spaceBetweenPipesMin, self.spaceBetweenPipesMax)
		self.spaceBetweenPipesHeight0 = random.randint(self.pipeHeight - self.spaceBetweenPipesMax / 2, self.pipeHeight + self.spaceBetweenPipesMin / 2)'''
		self.spaceBetweenPipes0=250
		self.spaceBetweenPipesHeight0=300
		self.spaceBetweenPipes1 = random.randint(int(self.spaceBetweenPipesMin), int(self.spaceBetweenPipesMax))
		self.spaceBetweenPipesHeight1 = random.randint(int(self.pipeHeight - self.spaceBetweenPipesMax / 2), int(self.pipeHeight + self.spaceBetweenPipesMin / 2))

		#Initialization Variables for Pipes
		self.pipeX0 = self.WIDTH / 2 - self.pipeWidth / 2 
		self.pipeX1 = self.WIDTH
		#self.pipeClosest = self.pipeX0

		#Initialization Variables for Scores
		self.flagPipePassed = 0
		self.flagScoresIncremented = 0
		self.maxScore=0

		#Create Flock (Standard Jump)
		'''for Bird in self.Birds:
			birdY = 200
			Bird.gravityAcceleration = 0
			#self.isJumping = 0
			#Bird.jumpHeight = 0

			#Initialization Variables for Life
			Bird.dead = 0
			Bird.birdX = 50
			Bird.birdY = 200
			Bird.birdPositionX = 0
			Bird.birdPositionY = 0

			#Initialization Variables for Score
			Bird.score=0
			Bird.flagUpdateScore = 0

			#Initialization number of jump made
			Bird.jumpMade=0'''

		#Create Flock (Parabolic Jump)
		for Bird in self.Birds:
			birdY = 200
			
			#Initialization Variables for Life
			Bird.dead = 0
			Bird.birdX = 50
			Bird.birdY = 200
			Bird.birdPositionX = 0
			Bird.birdPositionY = 0

			#Initialization Variables for Score
			Bird.lastScore=Bird.score
			Bird.score=0
			Bird.flagUpdateScore = 0

			#Initialization number of jump made
			Bird.jumpMade=0
			Bird.vy=0

		self.birdsAlive = [bird for bird in self.Birds if bird.dead == 0]

	def run(self):
		clock = pygame.time.Clock()	#Imports clock of the game
		pygame.font.init()
		font = pygame.font.SysFont("Arial", 25)
		#Games Loop
		while len(self.birdsAlive) != 0 and self.maxScore<self.maxFinalScore:
			#Controls the speed of the game
			clock.tick_busy_loop(self.clockSpeed)
			self.birdsPositions()

			#Event Handler
			for Bird in self.Birds:
				Bird.isJumping = 0
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					sys.exit()
			
			#Update position of the closest pipe
			if (self.pipeX0 < self.pipeX1):
				self.pipeClosest = self.pipeX0
			else:
				self.pipeClosest = self.pipeX1
			
			#GUI
			self.gui()

			#Increase of the difficulty
			self.difficultyIncrease()

			#Updating Birds
			self.updateScores()
			self.birdsAlive = [bird for bird in self.Birds if bird.dead == 0]
			
			self.updatePipes()

			self.nextSpaceBetweenPipes = self.getNextSpaceBetweenPipes()
			for bird in self.birdsAlive:
				self.screen.blit(bird.birdImgs[0], (bird.birdX, bird.birdY))
				bird.updateBird(self.nextSpaceBetweenPipes)
				bird.distanceTraveled += self.pipeSpeed
			self.updateLife()

			#Writing on file
			if self.maxScore>=100:
				self.writeOnFile(self.geneticCycles)

			#Refresh Screen
			pygame.display.update()

		#Game over
		self.screen.fill((255, 255, 255))
		self.screen.blit(self.gameOver, (40 + self.WIDTH/4, 250))
		self.index = 1
		self.interlineSCORES = 30
		for Bird in self.Birds:
			self.screen.blit(font.render(str(self.index) + " : " + str(Bird.distanceTraveled), -1, (0, 0, 0)), (self.WIDTH + 30, self.interlineSCORES))
			self.index += 1
			self.interlineSCORES += 50

	def geneticEvolution (self):
		self.genetic=GeneticAlgorithm(self.Birds)
		self.geneticCycles += 1
		self.Birds=self.genetic.runGeneticAlgorithm()

	def writeOnFile (self,nepochs):
		out_file = open("data/trainingDataset_Parabolic3"+str(nEpochs)+".txt","a")
		if len(self.birdsAlive)>0:
			out_file.write("%f," % self.birdsAlive[0].birdPositionX)
			out_file.write("%f," % self.birdsAlive[0].birdPositionY)
			out_file.write("%f," % self.nextSpaceBetweenPipes)
			out_file.write("%f\n" % self.birdsAlive[0].nn.takeDecision(self.birdsAlive[0].birdPositionX, self.birdsAlive[0].birdPositionY, self.nextSpaceBetweenPipes))
			out_file.close()

	def loadLearnedNN(self):
		#learnedNN = LearningAlgorithm()
		#self.Birds = []
		#self.Birds.append(Bird())

		l1 = LearningAlgorithm()
		self.Birds[len(self.Birds)-1].nn = l1.returnTrainedNN()
		self.Birds[len(self.Birds)-1].birdImgs = [pygame.image.load("img/redBirdFlap0.png").convert_alpha()]