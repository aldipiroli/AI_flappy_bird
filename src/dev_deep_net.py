from __future__ import division
import numpy as np
import random
import tensorflow as tf
from neuralNetwork import NeuralNetwork
# ---------------------------------------------------------------- #
# --- LearningAlgorithm() --- #
class LearningAlgorithm():
	def __init__(self):
		# ---------------------------------------------------------------- #
		# PARAMETERS:
		# Data Parameters:
		batch_size = 100

		# Learning Parameters:  
		LEARNING_RATE = 0.001
		LEARNING_THRESHOLD = 0.0001
		N_SCREEN_ITERATIONS = 1

		# Network Parameters:	   
		nInputs = 2
		nNeuronsLayer1 = 100
		nOutputs = 1

		# Network Construction:
		x_ = tf.placeholder(tf.float32, shape=[None, nInputs])
		y_ = tf.placeholder(tf.float32, shape=[None, nOutputs])

		W1 = tf.Variable(tf.truncated_normal([nInputs,nNeuronsLayer1], stddev=0.1))
		W2 = tf.Variable(tf.truncated_normal([nNeuronsLayer1,nNeuronsLayer1], stddev=0.1))
		W3 = tf.Variable(tf.truncated_normal([nNeuronsLayer1,nNeuronsLayer1], stddev=0.1))
		WOut = tf.Variable(tf.truncated_normal([nNeuronsLayer1,nOutputs], stddev=0.1))


		b1 = tf.Variable(tf.truncated_normal([nNeuronsLayer1], stddev=0.1))
		b2 = tf.Variable(tf.truncated_normal([nOutputs], stddev=0.1))
		b3 = tf.Variable(tf.truncated_normal([nOutputs], stddev=0.1))
		bOut = tf.Variable(tf.truncated_normal([nOutputs], stddev=0.1))


		y1 = tf.sigmoid(tf.matmul(x_, W1) + b1)
		y2 = tf.sigmoid(tf.matmul(y1, W2) + b2)
		y3 = tf.sigmoid(tf.matmul(y2, W3) + b3)
		
		yOut = tf.sigmoid(tf.matmul(y3, WOut) + bOut)

		# ---------------------------------------------------------------- #
		# Define loss, optimizer, accuracy:
		#cost = tf.reduce_mean(tf.nn.l2_loss(y_ - yOut))
		cost = tf.reduce_mean(tf.square(y_ - yOut)) 
		train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

		acc, acc_op = tf.metrics.accuracy(labels=tf.round(y_), predictions=tf.round(yOut))


		# ---------------------------------------------------------------- #
		# Session Initialization:
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()


		# ---------------------------------------------------------------- #
		# Prepare Data:
		datasetFeatures, datasetTargets = self.getTrainingSet()
		batchesLeft = int(len(datasetTargets)/batch_size)

		# ---------------------------------------------------------------- #
		# Train:
		i = 0
		k = 0
		while True:
			if batchesLeft > 0:
				if i % batch_size == 0:
					batch_x, batch_y, batchesLeft = self.getNextBatch(batch_size, i, batchesLeft,datasetFeatures,datasetTargets)
					out_batch, _ = sess.run((yOut, train_step), feed_dict={x_: batch_x, y_: batch_y})
			else:
				batchesLeft = int(len(datasetTargets)/batch_size)
				i = -1

			# Print Test:
			if k % 1000 == 0:
				v = sess.run([acc, acc_op], feed_dict={yOut: out_batch, y_: batch_y})					
				print("Accuracy:")
				print(v[0])
			
			# Condition On Exit:
			if v[0] > 0.90:
				break 

			k +=1
			i +=1

		print("Exited")
		# ---------------------------------------------------------------- #
		# Return Trained Neural Network 
		self.nnTrained = NeuralNetwork()
		self.nnTrained.W1 = sess.run(W1)
		self.nnTrained.W2 = sess.run(W2)
		self.nnTrained.b1 = sess.run(b1)
		self.nnTrained.b2 = sess.run(b2)
		self.returnTrainedNN()
			
# ---------------------------------------------------------------- #
# --- getTrainingSet() --- #
	def getTrainingSet(self):
		arrayShuffle = []
		datasetFeatures = []
		datasetTargets = []

		countJumps = 0
		countNoJumps = 0

		with open('data/data.txt', "rt") as f:
			for line in f:
				x1 = line.split(',')[0]
				x2 = line.split(',')[1]

				y1 = line.split(',')[2]
				#y1 = y1[:-1] # get rid of \n
				y1 = float(y1.replace(",", "."))
				#arrayShuffle.append([x1,x2,y1])
				
				if y1 > 1/2:
					for _ in range(0,10):
						x1 = float(x1) + self.getRandomNumber(0.02, -0.02)
						x2 = float(x2) + self.getRandomNumber(0.02, -0.02)
						y1 = 1
						countJumps +=1
						arrayShuffle.append([x1,x2,y1])
				else:
					countNoJumps +=1
					y1 = 0
				arrayShuffle.append([x1,x2,y1])
				
		random.shuffle(arrayShuffle)
		for j in range(0,len(arrayShuffle)):
			datasetFeatures.append([arrayShuffle[j][0],arrayShuffle[j][1]])
			datasetTargets.append([arrayShuffle[j][2]])

		print("Jump:",countJumps, "No Jumps:", countNoJumps)
		return datasetFeatures, datasetTargets

	def getNextBatch(self,batch_size, i, batchesLeft,datasetFeatures,datasetTargets):	
		batch_x = []
		batch_y = []
		batchesLeft -=1 
		for j in range(i,i+batch_size):
			batch_x.append(datasetFeatures[j])
			batch_y.append(datasetTargets[j])

		return batch_x, batch_y, batchesLeft

	def getRandomNumber(self, maxValue, minValue):
		return (np.random.random() * (-maxValue + minValue)) + maxValue

	def returnTrainedNN(self):
		return self.nnTrained

#if __name__ == "__main__":
#	LearningAlgorithm()