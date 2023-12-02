import os
from os.path import exists
from PIL import Image
import seaborn as sn
import numpy as np
import pickle
import pandas as pd
import platform
import matplotlib.pyplot as plt
from classifier import Classifier
from emotion_data import EmotionData

FILE_MODEL = "model.dat"
DATASET_TRAIN = "./dataset/train"
DATASET_TEST = "./dataset/test"
LEARNING_RATE = 1E-3
NB_ITERATION = 5
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
DATASET_FILE = "dataset.csv"
DO_LEARN = False

def write_header_dataset():
    line = []
    for i in range(1, 49):
        for j in range(1, 49):
            h = "px" + str(i * j)
            line.append(h)
    line.append("emotion")
    with open(DATASET_FILE, "w") as fd:
        fd.write(','.join(line))


def write_dataset_file(dir):
    with open(DATASET_FILE, "a") as fd:
        for i in range(len(EMOTIONS)):
            emotion = EMOTIONS[i]
            path = dir + "/" + emotion
            filenames = os.listdir(path)
            for fn in filenames:
                with Image.open(path + "/" + fn) as img:
                    em = EmotionData(emotion, img)
                    csv = [str(value) for value in em.to_array()]
                    csv.append(str(i))
                    fd.write(','.join(csv) + "\n")

def get_data():
    data = np.loadtxt(DATASET_FILE, skiprows=1, delimiter=',')
    inputs = data[:, :- 1] / 255
    outputs = data[:, -1]
    return inputs, outputs



def one_hot_encoder(value, values):
    return [1 if x == value else 0 for x in values]


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


def predict(inputs, outputs):
    predictions = []
    for i in range(len(inputs)):
        predictions.append(classifier.run(inputs[i]))
    #matrix = confusion_matrix(outputs, predictions)
    # df = pd.DataFrame(matrix, range(len(EMOTIONS)), range(len(EMOTIONS)))
    # sn.heatmap(df, annot=True, annot_kws={"size": 5})  # font size
    plt.show()
    return predictions



if __name__=="__main__":
    # os.remove(DATASET_FILE)
    # write_header_dataset()
    # write_dataset_file(DATASET_TRAIN)
    print(platform.architecture()[0])
    inputs, outputs = get_data()
    if not exists(FILE_MODEL) or DO_LEARN:
        classifier = Classifier(len(inputs[0]), len(EMOTIONS), LEARNING_RATE)
        learning(classifier, inputs, outputs)
        save_model(classifier)
    else:
        classifier = load_model()
    predictions = predict(inputs, outputs)