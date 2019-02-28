#import tensorflow as tf

import math
import random
import numpy as np

class NeuralNetwork:
    def __init__(self):
        #print("Init:")

        # NN structure definition
        self.nInputLayer = 2
        
        self.nNeuronsHiddenLayer1 = 100
        self.nNeuronsHiddenLayer2 = 100 
        self.nNeuronsHiddenLayer3 = 100

        self.nNeuronsOutputLayer = 1 

        self.maxValue = 2
        self.minValue = -2

        self.x = np.zeros((1,self.nInputs), dtype=np.float32)
        #Hidden layer
        self.W1 = (np.random.random(size = (self.nInputs,self.nNeuronsLayer1)) * (-self.maxValue + self.minValue)) + self.maxValue
        self.b1 = (np.random.random(size =(1,self.nNeuronsLayer1)) * (-self.maxValue + self.minValue)) + self.maxValue
        
        self.y1 = (np.random.random(size =(1,self.nNeuronsLayer1)) * (-self.maxValue + self.minValue)) + self.maxValue

        #Output Layer
        self.W2 = (np.random.random(size =( self.nNeuronsLayer1,1)) * (-self.maxValue + self.minValue)) + self.maxValue
        self.b2 = (np.random.random(size =( 1,self.nNeuronsLayer2)) * (-self.maxValue + self.minValue)) + self.maxValue

        # Debug
        self.yesJump = 0
        self.noJump = 0


    def normalizeX(self, posX):    #Normalize into [-10, 10] ##A Questo metodo non dovrebbe esser passato self tra i parametri, ma senno' non funziona

        # 0 < X < 432 
        #normPosX = ((Input - InputLow) / (InputHigh - InputLow)) * (OutputHigh - OutputLow) + OutputLow;
        normPosX = (posX / 432) * (10 + 10) - 10
        return normPosX

    def normalizeY(self, posY):    #Normalize into [-10, 10]

        # -270 < Y < 358
        #normPosX = ((Input - InputLow) / (InputHigh - InputLow)) * (OutputHigh - OutputLow) + OutputLow;
        normPosY = ((posY + 270) / (358 + 270)) * (10 + 10) - 10
        return normPosY

    def takeDecision(self, feature1, feature2):

        self.x[0][0] = feature1
        self.x[0][1] = feature2
        #self.x[0][2] = feature3



        self.y1 = np.matmul(self.x,self.W1) + self.b1
        
       
        for i in range(0, len(self.y1)): 
            #self.y1[i][0] = 1 / (1 + math.exp(-self.y1[i][0])) 
            self.y1[i] = np.tanh(self.y1[i])

            #sigmoid
        #print("y1:")
        #print(self.y1)

        self.y2 = np.matmul(self.y1,self.W2) + self.b2
        self.y2 = 1 / (1 + math.exp(-self.y2)) #sigmoid
        #self.y2 = np.tanh(self.y2)

       # print(self.y2)

        #print(self.y1)
        #print("y2:")
        #print(self.y2)
        '''if self.y2 > 0:
            print "Jump:"
            print(self.y2)
            self.yesJump += 1
        else:
            print "Don't Jump:"
            print(self.y2)
            self.noJump += 1'''

        return self.y2
    # --- End class crossOver --- 