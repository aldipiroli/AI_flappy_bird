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
		# Learning Parameters:  
		LEARNING_RATE = 0.01
		ACCURACY_ON_BATCH = 0.67

		# Network Parameters:
		nInputLayer = NeuralNetwork().nInputLayer
		nOutputLayer = NeuralNetwork().nOutputLayer

		'''nHiddenLayer1 = NeuralNetwork().nHiddenLayer1
		nHiddenLayer2 = NeuralNetwork().nHiddenLayer2
		nHiddenLayer3 = NeuralNetwork().nHiddenLayer3'''
		nHiddenLayer1 = 3
		nHiddenLayer2 = 2
		nHiddenLayer3 = 1


		stddev_ = 1

		# Network Construction:

		x_ = tf.placeholder(tf.float32, shape=[None, nInputLayer])
		y_ = tf.placeholder(tf.float32, shape=[None, nOutputLayer])

		W1 = tf.Variable(tf.random_normal([nInputLayer,nHiddenLayer1], stddev=stddev_))
		W2 = tf.Variable(tf.random_normal([nHiddenLayer1,nHiddenLayer2], stddev=stddev_))
		W3 = tf.Variable(tf.random_normal([nHiddenLayer2,nHiddenLayer3], stddev=stddev_))
		WOut = tf.Variable(tf.random_normal([nHiddenLayer3,nOutputLayer], stddev=stddev_))


		b1 = tf.Variable(tf.random_normal([nHiddenLayer1], stddev=stddev_))
		b2 = tf.Variable(tf.random_normal([nHiddenLayer2], stddev=stddev_))
		b3 = tf.Variable(tf.random_normal([nHiddenLayer3], stddev=stddev_))
		bOut = tf.Variable(tf.random_normal([nOutputLayer], stddev=stddev_))

		y1 = tf.sigmoid(tf.matmul(x_, W1) + b1)
		y2 = tf.sigmoid(tf.matmul(y1, W2) + b2)
		y3 = tf.sigmoid(tf.matmul(y2, W3) + b3)
		yOut = tf.sigmoid(tf.matmul(y3, WOut) + bOut)


		# ---------------------------------------------------------------- #
		# Define loss, optimizer, accuracy:
		#cost = tf.reduce_mean(tf.nn.l2_loss(y_ - yOut))
		#cost = tf.reduce_mean(tf.square(y_ - yOut)) 
		cost = -tf.reduce_sum(y_*tf.log(yOut) + (1 - y_)*tf.log(1-yOut))
		
		train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
		#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

		acc, acc_op = tf.metrics.accuracy(labels=tf.round(y_), predictions=tf.round(yOut))
		mean_absolute_error, update_op = tf.metrics.mean_absolute_error(labels=tf.round(y_), predictions=tf.round(yOut))

		# ---------------------------------------------------------------- #
		# Session Initialization:
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()


		# ---------------------------------------------------------------- #
		# Prepare Data:
		trainingFeatures, trainingTargets, testFeatures, testTargets, trainingDim, testDim = self.getTrainingSet()

		# Data Parameters:
		batch_size_training = int(trainingDim / 10)
		batch_size_test = testDim



		batchesLeftTraining = int(len(trainingTargets)/batch_size_training)
		batchesLeftTest = int(len(testTargets)/batch_size_test)

		# ---------------------------------------------------------------- #
		# Train:

		error_on_test_k0 = -1
		error_on_test_k1 = -1
		error_on_training_k0 = -1
		error_on_training_k1 = -1
		
		epoch_count = 0

		i = 0
		nEpochs = 0
		endOfTrainig = 0
		while endOfTrainig == 0:
			if batchesLeftTraining > 0:
				if i % batch_size_training == 0:
					batch_x, batch_y, batchesLeftTraining = self.getNextBatch(batch_size_training, i, batchesLeftTraining,trainingFeatures,trainingTargets, 0)
					out_batch, _ = sess.run((yOut, train_step), feed_dict={x_: batch_x, y_: batch_y})
					i += batch_size_training

			else:
				
				accTraining = sess.run([acc, acc_op], feed_dict={yOut: out_batch, y_: batch_y})					
				errTraining = sess.run([mean_absolute_error, update_op], feed_dict={yOut: out_batch, y_: batch_y})
				
				j = 0
				while batchesLeftTest > 0:
					if j % batch_size_test == 0:

						batch_x, batch_y, batchesLeftTest = self.getNextBatch(batch_size_test, j, batchesLeftTest,testFeatures,testTargets, 1)
						out_batch= sess.run((yOut), feed_dict={x_: batch_x, y_: batch_y})

						accTest = sess.run([acc, acc_op], feed_dict={yOut: out_batch, y_: batch_y})					
						errTest = sess.run([mean_absolute_error, update_op], feed_dict={yOut: out_batch, y_: batch_y})
						
						print("Acc Training:", accTraining[0], "Acc Test:", accTest[0])
						print("Err Training:", errTraining[0], "Err Test:", errTest[0])
						print("k0:", np.abs(error_on_test_k0 - error_on_training_k0), 'k1', np.abs(errTest[0] - errTraining[0]))
						print("Epoch: ",nEpochs)

						#if accTest[0] > 0.85:
						#	endOfTrainig = 1
						if errTest[0] < 0.10:
							endOfTrainig = 1

						#if nEpochs > 5 and (np.abs(error_on_test_k0 - error_on_training_k0) < np.abs(errTest[0] - errTraining[0]) - 0.0005):
						#	endOfTrainig = 1
				j +=1

				batchesLeftTraining = int(len(trainingTargets)/batch_size_training)
				batchesLeftTest = int(len(testTargets)/batch_size_test)

				#i = -1
				i = 0
				nEpochs += 1
				error_on_test_k0 = errTest[0]
				error_on_training_k0 = errTraining[0]


			#i +=1

		print("Exited")
		# ---------------------------------------------------------------- #
		# Return Trained Neural Network 
		self.nnTrained = NeuralNetwork()
		self.nnTrained.W1 = sess.run(W1)
		self.nnTrained.W2 = sess.run(W2)
		self.nnTrained.W3 = sess.run(W3)
		self.nnTrained.WOut = sess.run(WOut)


		self.nnTrained.b1 = sess.run(b1)
		self.nnTrained.b2 = sess.run(b2)
		self.nnTrained.b3 = sess.run(b3)
		self.nnTrained.bOut = sess.run(bOut)

		#np.savetxt('weights/t2.txt', self.nnTrained.W1, delimiter='\n')
		'''np.savetxt('weights/t1.txt', self.nnTrained.W2, delimiter=',')
		np.savetxt('weights/t1.txt', self.nnTrained.W3, delimiter=',')
		np.savetxt('weights/t1.txt', self.nnTrained.WOut, delimiter=',')

		np.savetxt('weights/t1.txt', self.nnTrained.b1, delimiter=',')
		np.savetxt('weights/t1.txt', self.nnTrained.b2, delimiter=',')
		np.savetxt('weights/t1.txt', self.nnTrained.b3, delimiter=',')
		np.savetxt('weights/t1.txt', self.nnTrained.bOut, delimiter=',')'''




		self.returnTrainedNN()
			
# ---------------------------------------------------------------- #
# --- getTrainingSet() --- #
	def getTrainingSet(self):
		datasetShuffle = []

		trainingFeatures = []
		trainingTargets = []

		testFeatures = []
		testTargets = []

		countJumps = 0
		countNoJumps = 0

		with open('data/trainingDataset_Parabolic2.txt', "rt") as f:
			for line in f:
				x1 = line.split(',')[0]
				x2 = line.split(',')[1]
				x3 = line.split(',')[2]

				y1 = line.split(',')[3]
				#y1 = y1[:-1] # get rid of \n
				y1 = float(y1.replace(",", "."))
				#datasetShuffle.append([x1,x2,y1])
				
				if y1 > 1/2:
					#x1 = int(x1)
					#x2 = int(x2)
					y1 = 1
					datasetShuffle.append([x1,x2,x3,y1])
					for _ in range(0, 3):
						#x1 = float(x1) + self.getRandomNumber(0.005, -0.005)
						#x2 = float(x2) + self.getRandomNumber(0.005, -0.005)
						#x1 = int(x1)
						#x2 = int(x2)
						y1 = 1
						countJumps +=1
						datasetShuffle.append([x1,x2,x3,y1])
				else:
					countNoJumps +=1
					#x1 = int(x1)
					#x2 = int(x2)
					y1 = 0
					datasetShuffle.append([x1,x2,x3,y1])
				
		random.shuffle(datasetShuffle)

		trainingDim = int(len(datasetShuffle )*0.8)
		testDim = (len(datasetShuffle) - trainingDim)
		for j in range(0,trainingDim):
			trainingFeatures.append([datasetShuffle[j][0],datasetShuffle[j][1],datasetShuffle[j][2]])
			trainingTargets.append([datasetShuffle[j][3]])

		for j in range(trainingDim,len(datasetShuffle)):
			testFeatures.append([datasetShuffle[j][0],datasetShuffle[j][1],datasetShuffle[j][2]])
			testTargets.append([datasetShuffle[j][3]])
		
		print("Jump:",countJumps, "No Jumps:", countNoJumps)
		return trainingFeatures, trainingTargets, testFeatures, testTargets, trainingDim, testDim


	def getNextBatch(self,batch_size, i, batchesLeft, features, targets, isTest):	
		batch_x = []
		batch_y = []
		batchesLeft -=1 

		if len(features) < batch_size:
			batch_size = len(features)

		for j in range(i,i+batch_size):
			if isTest == 1:
				#batch_x.append([features[j][0]+self.getRandomNumber(0.1, -0.1), features[j][1]+self.getRandomNumber(0.1, -0.1)])
				batch_x.append(features[j])
				batch_y.append(targets[j])
			else:
				batch_x.append(features[j])
				batch_y.append(targets[j])

		return batch_x, batch_y, batchesLeft

	def getRandomNumber(self, maxValue, minValue):
		return (np.random.random() * (-maxValue + minValue)) + maxValue

	def returnTrainedNN(self):
		return self.nnTrained

#if __name__ == "__main__":
#	LearningAlgorithm()