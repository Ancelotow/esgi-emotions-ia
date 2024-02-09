from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import os
from os.path import exists
from PIL import Image
import platform
import cv2

from api.lib.dataset import write_header_dataset, write_dataset_file, rescale_image
from api.lib.machine_learning import get_data, load_model, save_model

PATH = "eyes"
# DATASETS
FILE_MODEL = "../../dataset/" + PATH + "/model.dat"
DATASET_TRAIN = "../../dataset/" + PATH + "/train"
DATASET_TEST = "../../dataset/" + PATH + "/test"
DATASET_FILE = "../../dataset/" + PATH + "/dataset_test.csv"

DATASET_NAME_TEST = "../../dataset/" + PATH + "/dataset_train.csv"
DATASET_NAME_TRAIN = "../../dataset/" + PATH + "/dataset_train.csv"
DATASET_NAME_PREDICT = "../../dataset/" + PATH + "/dataset_predict.csv"

# CLASSIFICATION
CLASSIFICATION = ["amber", "blue", "brown", "gray", "grayscale", "green", "hazel", "red"]

# LEARNING
DO_LEARN = True
HIDDEN_LAYER_SIZE = (150, 250, 350, 250, 150)

if __name__ == '__main__':
    # os.remove(DATASET_FILE)
    # Creation dataset.csv
    # rescale_image(DATASET_TEST, 27)
    # write_header_dataset("eyes", 27, DATASET_FILE)
    # write_dataset_file(DATASET_TEST, CLASSIFICATION, DATASET_FILE)

    print(os.getcwd())
    inputs, outputs = get_data("TEST", PATH)
    if not exists(FILE_MODEL) or DO_LEARN:
        # TRAIN
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=100)
        classifier.fit(inputs, outputs)
        save_model(classifier, FILE_MODEL)
    else:
        classifier = load_model(FILE_MODEL)

    predict = classifier.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    score = accuracy_score(outputs, predict)
    color = 'white'
    disp = ConfusionMatrixDisplay(confusion_matrix=margin_errors, display_labels=CLASSIFICATION)
    disp.plot()
    plt.show()
    print("Score : " + str(score) + "\n")

    # PREDICT
    #inputs = []
    #with Image.open("../../dataset/predict/eyes.png") as img:
    #    for x in range(img.width):
    #        for y in range(img.height):
    #            inputs.append(img.getpixel((x, y)))
    #predict = classifier.predict([inputs])
    #predict_eyes = CLASSIFICATION[int(predict[0])]
    #print("EYES Predict : " + str(predict_eyes))
