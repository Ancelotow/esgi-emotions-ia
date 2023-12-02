from math import *
from random import random

class Neuron:
    def __init__(self, inputs, learning_rate):
        self.weights = []
        self.learning_rate = learning_rate
        for i in range(inputs):
            self.weights.append(random() * 2 + -1)

    def run(self, inputs):
        output = 0
        for i in range(len(self.weights)):
            output += self.weights[i] * inputs[i]
        output = 1.0 / (1.0 + exp(- output))
        return output

    def learn(self, inputs, desired):
        output = self.run(inputs)
        error = desired - output
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        return error