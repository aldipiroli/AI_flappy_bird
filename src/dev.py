from neuralNetwork import NeuralNetwork
from bird import Bird
import numpy as np
import sys

# --- GeneticAlgorithm() --- 
class GeneticAlgorithm():
	def __init__(self, populationArray):
		# ----------------------------------------------------------------------- # 
		# Population:
		nPopulation = len(populationArray)

		nNotChanged = int(nPopulation / 4)
		nCrossed =  int(nPopulation / 4) 
		nMutated = int(nPopulation - (nNotChanged + nCrossed))

		print("nPopulation:",nPopulation,"nNotChanged", nNotChanged, "nCrossed", nCrossed, "nMutated", nMutated)

		if((nNotChanged + nCrossed + nMutated) != nPopulation):
			sys.exit("Error: Total number of the new population doesn't match with previous")

		# Network:
		nWeightsToMutateHiddenLayer1 = self.getRandomNumberInteger(populationArray[0].nn.nHiddenLayer1,1)
		nWeightsToMutateHiddenLayer2 = self.getRandomNumberInteger(populationArray[0].nn.nHiddenLayer2,1)
		nWeightsToMutateHiddenLayer3 = self.getRandomNumberInteger(populationArray[0].nn.nHiddenLayer3,1)

		mutationScale = 1


	def rankPopulation(self, populationArray):
		populationArray = sorted(populationArray, key=lambda populationArray: populationArray.jumpMade)
		populationArray = sorted(populationArray, key=lambda populationArray: populationArray.distanceTraveled, reverse=True)
		return populationArray 

	def crossover(self, nNotChanged, nCrossed):
		for i in range(nNotChanged, nNotChanged + nCrossed): 
			#crossoverPoint = self.getRandomNumberInteger(populationArray[i].nn.nNeuronsLayer1,0)
			crossoverPoint = 1

			populationArray[i].nn = self.crossDNA(populationArray[i-nNotChanged].nn, populationArray[(i+1)-nNotChanged].nn)
			

	def crossDNA(self, parent1, parent2, crossoverPoint):
		child = NeuralNetwork()







		for i in range(0, parent1.nInputs):
			for j in range(0, crossoverPoint):
				child.W1[i][j] = parent1.W1[i][j]

		for i in range(0, parent1.nInputs):
			for j in range(0, crossoverPoint):
				child.W1[i][j] = parent2.W1[i][j]
		
		#NB: 0 here is because W2 [_,_,_]  
		for i in range(0, crossoverPoint):
			child.W2[i][0] = parent1.W2[i][0]

		for i in range(crossoverPoint, parent2.nNeuronsLayer1):
			child.W2[i][0] = parent2.W2[i][0]
		
		return child			


	def mutation(self):
		for i in range(0, nNotChanged): 
			mutatedNN = NeuralNetwork()
			mutatedNN = populationArray[i].nn
			for j in range(0, numberOfWeightsToMutate):
				_j = self.getRandomNumberInteger(mutatedNN.nNeuronsLayer1,0)
				_i = self.getRandomNumberInteger(mutatedNN.nInputs,0)
				_mutationValue = self.getRandomNumber(self.mutationScale, -self.mutationScale)
				#_mutationValue = 0

				mutatedNN.W1[_i][_j] = mutatedNN.W1[_i][_j] + _mutationValue

			index = int(nPopulation - (i+1))
			#print("INDEX:", index,nPopulation)
			populationArray[index].nn = mutatedNN
		
	def getRandomNumber(self, maxValue, minValue):
		return (np.random.random() * (-maxValue + minValue)) + maxValue

	def getRandomNumberInteger(self, maxValue, minValue):
		return (np.random.randint(minValue, maxValue))

	def runGeneticAlgorithm(self):
		self.rankPopulation()
		self.crossover()
		self.mutation()
		return populationArray



if __name__ == "__main__":
	populationArray = []

	populationArray.append(NeuralNetwork())
	populationArray.append(NeuralNetwork())
	populationArray.append(NeuralNetwork())

	GeneticAlgorithm(populationArray)