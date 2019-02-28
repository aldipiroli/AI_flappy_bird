from __future__ import division
from neuralNetwork import NeuralNetwork
from bird import Bird
import numpy as np
import sys

# --- GeneticAlgorithm() --- 
class GeneticAlgorithm():
	def __init__(self, populationArray):
		self.populationArray = populationArray

		# ----------------------------------------------------------------------- # 
		# Define Stucture of new population
		# Population:
		self.nPopulation = len(self.populationArray)
		self.nNotChanged = int(self.nPopulation / 4)
		self.nCrossed =  int(self.nPopulation / 4) 
		self.nMutated = int(self.nPopulation - (self.nNotChanged + self.nCrossed))

		if((self.nNotChanged + self.nCrossed + self.nMutated) != self.nPopulation):
			sys.exit("Error: Total number of the new population doesn't match with previous")

		# Network:
		self.numberOfWeightsToMutate_1 = self.getRandomNumberInteger(self.populationArray[0].nn.nHiddenLayer1,1)
		self.numberOfWeightsToMutate_2 = self.getRandomNumberInteger(self.populationArray[0].nn.nHiddenLayer2,1)
		self.numberOfWeightsToMutate_3 = self.getRandomNumberInteger(self.populationArray[0].nn.nHiddenLayer3,1)
		
		#self.mutationScale = self.getRandomNumber(self.populationArray[0].nn.maxValue, self.populationArray[0].nn.minValue)
		self.mutationScale = self.getRandomNumber(1,-1)
		#self.mutationScale = 0.5
		# ----------------------------------------------------------------------- # 


	#def printPopulationArray(self):
		#print("---")
		#for i in range(0, self.nPopulation):
			#print(self.populationArray[i].nn.W1)
		#print("---")
		#sys.exit(0)

	def rankPopulation(self):
		self.populationArray = sorted(self.populationArray, key=lambda populationArray: populationArray.lastScore, reverse=True)
		self.populationArray = sorted(self.populationArray, key=lambda populationArray: populationArray.distanceTraveled, reverse=True) 

	'''def crossover(self):
		for i in range(self.nNotChanged + 2, self.nNotChanged + self.nCrossed + 2): 

			#print("INDEX CROSS:", i,self.nPopulation)
			#crossoverPoint = self.getRandomNumberInteger(self.populationArray[i].nn.nNeuronsLayer1,0)
			crossoverPoint = self.populationArray[i].nn.nNeuronsLayer1/2
			self.populationArray[i].nn = self.crossDNA(self.populationArray[i-self.nNotChanged].nn, self.populationArray[(i+1)-self.nNotChanged].nn, crossoverPoint)'''
	
	def crossover(self):	#Aldi
		for i in range(self.nNotChanged + 2, self.nNotChanged + self.nCrossed + 2): 
			#print("INDEX CROSS:", i,self.nPopulation)
			self.populationArray[i].nn = self.crossDNA(self.populationArray[i-self.nNotChanged].nn, self.populationArray[(i+1)-self.nNotChanged].nn)
			

	def crossDNA(self, parent1, parent2):
		child = NeuralNetwork()

		# ----------------------------------------------------------------------- # 
		# W1
		crossoverPoint = self.getRandomNumberInteger(parent1.nHiddenLayer1,0)
		for i in range(0, parent1.nInputLayer):
			for j in range(0, crossoverPoint):
				child.W1[i][j] = parent1.W1[i][j]
				child.b1[0][j] = parent1.b1[0][j]

		for i in range(0, parent1.nInputLayer):
			for j in range(crossoverPoint, parent1.nHiddenLayer1):
				child.W1[i][j] = parent2.W1[i][j]
				child.b1[0][j] = parent2.b1[0][j]
		
		# ----------------------------------------------------------------------- # 
		# W2
		crossoverPoint = self.getRandomNumberInteger(parent1.nHiddenLayer2,0)
		for i in range(0, parent1.nHiddenLayer1):
			for j in range(0, crossoverPoint):
				child.W2[i][j] = parent1.W2[i][j]
				child.b2[0][j] = parent1.b2[0][j]

		for i in range(0, parent1.nHiddenLayer1):
			for j in range(crossoverPoint, parent1.nHiddenLayer2):
				child.W2[i][j] = parent2.W2[i][j]
				child.b2[0][j] = parent2.b2[0][j]
		
		# ----------------------------------------------------------------------- # 
		# W3
		crossoverPoint = self.getRandomNumberInteger(parent1.nHiddenLayer3,0)
		for i in range(0, parent1.nHiddenLayer2):
			for j in range(0, crossoverPoint):
				child.W3[i][j] = parent1.W3[i][j]
				child.b3[0][j] = parent1.b3[0][j]

		for i in range(0, parent1.nHiddenLayer2):
			for j in range(crossoverPoint, parent1.nHiddenLayer3):
				child.W3[i][j] = parent2.W3[i][j]
				child.b3[0][j] = parent2.b3[0][j]
		
		# ----------------------------------------------------------------------- # 
		# WOut
		crossoverPoint = self.getRandomNumberInteger(parent1.nOutputLayer,0)	
		for i in range(0, parent1.nHiddenLayer3):
			for j in range(0, crossoverPoint):
				child.WOut[i][j] = parent1.WOut[i][j]
				child.bOut[0][j] = parent1.bOut[0][j]

		for i in range(0, parent1.nHiddenLayer3):
			for j in range(crossoverPoint, parent1.nOutputLayer):
				child.WOut[i][j] = parent2.WOut[i][j]
				child.bOut[0][j] = parent2.bOut[0][j]

		return child			


	def mutation(self):
		mutatedNN = NeuralNetwork()
		for i in range(0, self.nNotChanged): 
			mutatedNN = self.populationArray[i].nn
			for j in range(0, self.numberOfWeightsToMutate_1):
				_j = self.getRandomNumberInteger(mutatedNN.nHiddenLayer1,0)
				_i = self.getRandomNumberInteger(mutatedNN.nInputLayer,0)
				_mutationValue = self.getRandomNumber(self.mutationScale, -self.mutationScale)
				mutatedNN.W1[_i][_j] = mutatedNN.W1[_i][_j] + _mutationValue
				mutatedNN.b1[0][_j] = mutatedNN.b1[0][_j] + _mutationValue

			for j in range(0, self.numberOfWeightsToMutate_2):
				_j = self.getRandomNumberInteger(mutatedNN.nHiddenLayer2,0)
				_i = self.getRandomNumberInteger(mutatedNN.nHiddenLayer1,0)
				_mutationValue = self.getRandomNumber(self.mutationScale, -self.mutationScale)
				mutatedNN.W2[_i][_j] = mutatedNN.W2[_i][_j] + _mutationValue
				mutatedNN.b2[0][_j] = mutatedNN.b2[0][_j] + _mutationValue

			for j in range(0, self.numberOfWeightsToMutate_3):
				_j = self.getRandomNumberInteger(mutatedNN.nHiddenLayer3,0)
				_i = self.getRandomNumberInteger(mutatedNN.nHiddenLayer2,0)
				_mutationValue = self.getRandomNumber(self.mutationScale, -self.mutationScale)
				mutatedNN.W3[_i][_j] = mutatedNN.W3[_i][_j] + _mutationValue
				mutatedNN.b3[0][_j] = mutatedNN.b3[0][_j] + _mutationValue

			index = int(self.nPopulation - (i+1))
			#print("INDEX MUTATI:", index,self.nPopulation)
			self.populationArray[index].nn = mutatedNN

	def new(self):	
		for i in range(0, self.nNotChanged): 
			mutatedNN = NeuralNetwork()
			index = int(self.nPopulation - self.nNotChanged - (i+1))
			#print("INDEX NUOVI:", index,self.nPopulation)
			self.populationArray[index].nn = mutatedNN
		
	def getRandomNumber(self, maxValue, minValue):
		return (np.random.random() * (-maxValue + minValue)) + maxValue

	def getRandomNumberInteger(self, maxValue, minValue):
		return (np.random.randint(minValue, maxValue))

	def runGeneticAlgorithm(self):
		#self.printPopulationArray()
		self.rankPopulation()
		#self.printPopulationArray()
		self.crossover()
		self.mutation()
		self.new()
		#self.printPopulationArray()
		return self.populationArray