from neuron import Neuron


def argmax(values):
    index = 0
    for i in range(len(values)):
        if values[i] > values[index]:
            index = i
    return index


class Classifier:
    def __init__(self, inputs, classes, learning_rate):
        self.neurons = []
        for i in range(classes):
            self.neurons.append(Neuron(inputs, learning_rate))

    def learn(self, inputs, desired):
        error = 0
        for i in range(len(self.neurons)):
            error += self.neurons[i].learn(inputs, desired[i])
        return error / len(self.neurons)

    def run(self, inputs):
        outputs = []
        for n in self.neurons:
            outputs.append(n.run(inputs))
        return argmax(outputs)