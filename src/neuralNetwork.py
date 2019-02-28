#import tensorflow as tf
from __future__ import division
import math
import random
import numpy as np

class NeuralNetwork:
	def __init__(self):
		# NN structure definition
		self.nInputLayer = 3
		self.nOutputLayer = 1

		self.nHiddenLayer1 = 10
		self.nHiddenLayer2 = 7
		self.nHiddenLayer3 = 5

		self.maxValue = 2
		self.minValue = -2

		self.x = np.zeros((1,self.nInputLayer), dtype=np.float32)
		
		#Hidden layer
		self.W1 = (np.random.normal(0, 1, size = (self.nInputLayer,self.nHiddenLayer1)))
		self.W2 = (np.random.normal(0, 1, size = (self.nHiddenLayer1,self.nHiddenLayer2)))
		self.W3 = (np.random.normal(0, 1, size = (self.nHiddenLayer2,self.nHiddenLayer3)))
		self.WOut = (np.random.normal(0, 1, size = (self.nHiddenLayer3,self.nOutputLayer)))

		self.b1 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer1)))
		self.b2 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer2)))
		self.b3 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer3)))
		self.bOut = (np.random.normal(0, 1, size =(1,self.nOutputLayer)))

		self.y1 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer1)))
		self.y2 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer2)))
		self.y3 = (np.random.normal(0, 1, size =(1,self.nHiddenLayer3)))
		self.yOut = (np.random.normal(0, 1, size =(self.nOutputLayer,1)))


		# Debug
		self.yesJump = 0
		self.noJump = 0


	def takeDecision(self, feature1, feature2, feature3):

		self.x[0][0] = feature1
		self.x[0][1] = feature2
		self.x[0][2] = feature3

		self.y1 = np.matmul(self.x,self.W1) + self.b1		
		for i in range(0, len(self.y1)): 
			self.y1[0][i] = self.sigmoid(self.y1[0][i])

		self.y2 = np.matmul(self.y1,self.W2) + self.b2
		for i in range(0, len(self.y2)): 
			self.y2[0][i] = self.sigmoid(self.y2[0][i])

		self.y3 = np.matmul(self.y2,self.W3) + self.b3
		for i in range(0, len(self.y3)): 
			self.y3[0][i] = self.sigmoid(self.y3[0][i])

		self.yOut = self.sigmoid(np.matmul(self.y3,self.WOut) + self.bOut)

		if self.yOut > 1/2:
			return 1
		else:
			return 0

	# --- End class crossOver --- 
	def sigmoid(self, x):
		return float(1 / (1 + np.exp(-x)))