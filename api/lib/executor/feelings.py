import os
import cv2
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from os.path import exists
from skimage import io
from api.lib.machine_learning import load_model, save_model
import matplotlib.pyplot as plt
from skimage.transform import resize

PATH = "feelings"
# DATASETS
FILE_MODEL = "../../dataset/"+PATH+"/model.dat"
DATASET_TRAIN = "../../dataset/"+PATH+"/train"
DATASET_TEST = "../../dataset/"+PATH+"/test"

# CLASSIFICATION
CLASSIFICATION = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Parameters
DO_LEARN = True
HIDDEN_LAYER_SIZE = ()
MAX_ITER = 50
CURRENT_DATASET = DATASET_TRAIN
TYPE_DATASET = "TRAIN"
TRANSFORM_IMAGE = True

def update_images(dir):
    for i in range(len(CLASSIFICATION)):
        classification_folder = CLASSIFICATION[i]
        path = dir + "/" + classification_folder
        filenames = os.listdir(path)
        for fn in filenames:
            img_path = path + "/" + fn
            img = io.imread(img_path, as_gray=True)  # Load the image in grayscale
            img_resized = resize(img, (64, 64))
            for _ in range(2):
                sobelx = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 0, 1, ksize=5)
                img_resized = np.hypot(sobelx, sobely)
                img_resized *= 255.0 / np.max(img_resized)
            cv2.imwrite(img_path, img_resized)


def get_data(dir):
    inputs = []
    outputs = []
    for i in range(len(CLASSIFICATION)):
        classification_folder = CLASSIFICATION[i]
        path = dir + "/" + classification_folder
        filenames = os.listdir(path)
        for fn in filenames:
            img = io.imread(path + "/" + fn)
            img_resized = resize(img, (64, 64))
            inputs.append(img_resized.flatten().tolist())
            outputs.append(i)
    return inputs, outputs


if __name__ == '__main__':
    if TRANSFORM_IMAGE:
        print("Transform image...")
        update_images(CURRENT_DATASET)

    print("Getting data...")
    inputs, outputs = get_data(CURRENT_DATASET)
    if not exists(FILE_MODEL) or DO_LEARN:
        print("Learning...")
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=MAX_ITER)
        classifier.fit(inputs, outputs)
        save_model(classifier, FILE_MODEL)
    else:
        classifier = load_model(FILE_MODEL)

    print("Predicting...")
    predict = classifier.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    score = accuracy_score(outputs, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=margin_errors, display_labels=CLASSIFICATION)
    disp.plot()
    plt.show()
    print("Score : " + str(score) + "\n")

