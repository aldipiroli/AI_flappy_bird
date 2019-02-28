from __future__ import division
import numpy as np
import random
import tensorflow as tf

# ---------------------------------------------------------------- #
# --- LearningAlgorithm() --- #
class LearningAlgorithm():
	def __init__(self):
		# ---------------------------------------------------------------- #
		# PARAMETERS:
		# Data Parameters:
		batch_size = 300

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

		W2 = tf.Variable(tf.truncated_normal([nNeuronsLayer1,nOutputs], stddev=0.1))

		b1 = tf.Variable(tf.truncated_normal([nNeuronsLayer1], stddev=0.1))
		b2 = tf.Variable(tf.truncated_normal([nOutputs], stddev=0.1))

		y1 = tf.matmul(x_, W1) + b1
		y2 = tf.matmul(y1, W2) + b2

		# ---------------------------------------------------------------- #
		# Define loss, optimizer, accuracy:
		#cost = tf.reduce_mean(tf.nn.l2_loss(y_ - y2))
		cost = tf.reduce_mean(tf.square(y_ - y2)) 
		train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(y2), tf.int32),tf.cast(tf.round(y_), tf.int32)), tf.float32))

		# ---------------------------------------------------------------- #
		# Session Initialization:
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		# ---------------------------------------------------------------- #
		# Prepare Data:
		datasetFeatures, datasetTargets = self.getTrainingSet()
		batchesLeft = int(len(datasetTargets)/batch_size)

		# ---------------------------------------------------------------- #
		# Train:
		i = 0
		for k in range(1000000):
			if batchesLeft > 0:
				if i % batch_size == 0:
					batch_x, batch_y, batchesLeft = self.getNextBatch(batch_size, i, batchesLeft,datasetFeatures,datasetTargets)
					sess.run((train_step), feed_dict={x_: batch_x, y_: batch_y})
			else:
				batchesLeft = int(len(datasetTargets)/batch_size)
				i = -1

			
			# Test:
			if k % 10000== 0:
				out_batch, acc = sess.run((y2, accuracy), feed_dict={x_: batch_x, y_: batch_y})
				inx_ = 0
				#print(batch_x[inx_][0], " + ", batch_x[inx_][1], " = ", out_batch[inx_][0], "|", batch_y[inx_], "cost:",cost_)
				print("Network: ",out_batch[inx_][0], "Target: ", batch_y[inx_][0], "|", "acc:",acc*100, "%")


			
			i +=1
			k +=1
			
# ---------------------------------------------------------------- #
# --- getTrainingSet() --- #
	def getTrainingSet(self):
		arrayShuffle = []
		datasetFeatures = []
		datasetTargets = []

		with open('data/data2.txt', "rt") as f:
			for line in f:
				x1 = line.split(',')[0]
				x2 = line.split(',')[1]

				y1 = line.split(',')[2]
				#y1 = y1[:-1] # get rid of \n
				y1 = float(y1.replace(",", "."))
				arrayShuffle.append([x1,x2,y1])
				'''if y1 > 1/2:
					for _ in range(0,10):
						x1 = float(x1) + self.getRandomNumber(0.02, -0.02)
						x2 = float(x2) + self.getRandomNumber(0.02, -0.02)
						y1 = 1
						arrayShuffle.append([x1,x2,y1])
				else:
					y1 = 0
				arrayShuffle.append([x1,x2,y1])
				'''

		random.shuffle(arrayShuffle)
		for j in range(0,len(arrayShuffle)):
			datasetFeatures.append([arrayShuffle[j][0],arrayShuffle[j][1]])
			datasetTargets.append([arrayShuffle[j][2]])
		return datasetFeatures, datasetTargets

	def getNextBatch(self,batch_size, i, batchesLeft,datasetFeatures,datasetTargets):	
		batch_x = []
		batch_y = []
		batchesLeft -=1 
		for j in range(i,i+1):
			batch_x.append(datasetFeatures[j])
			batch_y.append(datasetTargets[j])

		return batch_x, batch_y, batchesLeft

	def getRandomNumber(self, maxValue, minValue):
		return (np.random.random() * (-maxValue + minValue)) + maxValue

if __name__ == "__main__":
	LearningAlgorithm()