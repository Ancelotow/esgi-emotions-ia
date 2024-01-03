import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from lib.graphs import one_hot_encoder
from model.neuron import Neuron

FILE_MODEL = "model.dat"
LEARNING_RATE = 1E-3
NB_ITERATION = 5
DATASET_NAME_TEST = "./dataset_train.csv"
DATASET_NAME_TRAIN = "./dataset_train.csv"
DATASET_NAME_PREDICT = "./dataset_predict.csv"


def get_data(type):
    if type == "TEST":
        filename = DATASET_NAME_TEST
    elif type == "TRAIN":
        filename = DATASET_NAME_TRAIN
    else:
        filename = DATASET_NAME_PREDICT
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    inputs = data[:, :- 1] / 255
    outputs = data[:, -1]
    return inputs, outputs


def learning(classifier, inputs, outputs):
    errors = []
    for j in range(NB_ITERATION):
        for i in range(len(inputs)):
            desired = one_hot_encoder(outputs[i], list(range(10)))
            errors.append(classifier.learn(inputs[i], desired))
    plt.plot(errors)
    plt.show()


def save_model(classifier):
    with open(FILE_MODEL, 'wb') as file:
        pickle.dump(classifier, file)


def load_model():
    with open(FILE_MODEL, 'rb') as file:
        classifier = pickle.load(file)
    return classifier


def predict(inputs, outputs, classifier):
    predictions = []
    for i in range(len(inputs)):
        predictions.append(classifier.run(inputs[i]))
    # matrix = confusion_matrix(outputs, predictions)
    # df = pd.DataFrame(matrix, range(len(EMOTIONS)), range(len(EMOTIONS)))
    # sn.heatmap(df, annot=True, annot_kws={"size": 5})  # font size
    plt.show()
    return predictions


def classification(inputs, outputs):
    n = Neuron(len(inputs[0]), 1E-7)
    errors = []
    for j in range(3000):
        for i in range(len(inputs)):
            errors.append(n.learn(inputs[i], outputs[i]))

    plt.plot(errors)
    plt.show()

    error = 0.0
    predictions = []
    for i in range(len(inputs)):
        prediction = n.run(inputs[i])
        print(prediction, ' -- ', outputs[i])
        predictions.append(prediction)
        error += abs(prediction - outputs[i])
    error = error / len(inputs)
    return n, predictions, error
