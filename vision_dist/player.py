import random
import numpy as np
import sys

def sigmoid(z):
    return (1.0 + np.exp(-z)) ** -1.0

def reLU(z):
    return z

class Player:
    def __init__(self,  weights = None, biases = None, sizes = [32,20,12,4]):
    #def __init__(self, sizes = [8,12,4]):
        self.num_layers = len(sizes)
        self.sizes = sizes

        if biases is not None:
          self.biases = biases
        else:
          self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        if weights is not None:
          self.weights = weights
        else:
          self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def forwardprop(self, activation):
        for bias, weight in zip(self.biases, self.weights):
           activation = reLU(np.dot(weight, activation) + bias)
        return sigmoid(activation)

    # direction values -1, 1 (up, down) or 2, 4 (left, right)
    def move(self, input):
        tmp = np.argmax(self.forwardprop(input))
        if tmp == 0 :
            tmp -= 1
        if tmp == 3 :
            tmp += 1
        return tmp
        
    def getNetwork(self):
      return self.weights, self.biases

