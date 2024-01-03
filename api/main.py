import pickle
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from lib.machine_learning import get_data, save_model, load_model, predict
import os
from os.path import exists
from PIL import Image
import platform
import matplotlib.pyplot as plt
import cv2

from model.model import Model

FILE_MODEL = "model.dat"
DATASET_TRAIN = "./dataset/train"
DATASET_TEST = "./dataset/test"
DATASET_PREDICT = "./dataset/predict"
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
DATASET_FILE = "dataset_train.csv"
DO_LEARN = False
HIDDEN_LAYER_SIZE = (300, 300)


def write_header_dataset():
    line = []
    for i in range(1, 49):
        for j in range(1, 49):
            h = "px" + str(i * j)
            line.append(h)
    line.append("emotion")
    with open(DATASET_FILE, "w") as fd:
        fd.write(','.join(line))
        fd.write("\n")


def write_dataset_file(dir):
    with open(DATASET_FILE, "a") as fd:
        for i in range(len(EMOTIONS)):
            emotion = EMOTIONS[i]
            path = dir + "/" + emotion
            filenames = os.listdir(path)
            for fn in filenames:
                with Image.open(path + "/" + fn) as img:
                    em = Model(emotion, img)
                    csv = [str(value) for value in em.to_array()]
                    csv.append(str(i))
                    fd.write(','.join(csv) + "\n")


if __name__ == '__main__':
    # os.remove(DATASET_FILE)
    # Creation dataset.csv
    # write_header_dataset()
    # write_dataset_file(DATASET_TEST)

    print(platform.architecture()[0])
    inputs, outputs = get_data("TRAIN")
    if not exists(FILE_MODEL) or DO_LEARN:
        # TRAIN
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=2000)
        classifier.fit(inputs, outputs)
        margin_errors = classifier.score(inputs, outputs)
        print("Margin of errors ", str(margin_errors))
        # plt.plot(classifier.loss_curve_, color="blue")
        # plt.show()
        save_model(classifier)
    else:
        classifier = load_model()

    predict = classifier.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    print("Errors : " + str(margin_errors))

    # PREDICT
    inputs = []
    with Image.open("./dataset/predict/face.png") as img:
        for x in range(img.width):
            for y in range(img.height):
                inputs.append(img.getpixel((x, y)))
    predict = classifier.predict([inputs])
    predict_feeling = EMOTIONS[int(predict[0])]
    print("Predict : " + str(predict_feeling))
