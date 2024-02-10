import os
import cv2
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
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
HIDDEN_LAYER = (10, 10)
MAX_ITER = 30
TRANSFORM_IMAGE = False

def update_images(dir):
    for i in range(len(CLASSIFICATION)):
        classification_folder = CLASSIFICATION[i]
        path = dir + "/" + classification_folder
        filenames = os.listdir(path)
        for fn in filenames:
            img_path = path + "/" + fn
            img = io.imread(img_path, as_gray=True)  # Load the image in grayscale
            img_resized = resize(img, (128, 128))

            # Apply Sobel filter
            sobel_x = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            sobel = (sobel * 255.0 / np.max(sobel)).astype(np.uint8)  # Convert to uint8

            # Apply dilation
            kernel = np.ones((3, 3), np.uint8)
            img_resized = cv2.dilate(sobel, kernel, iterations=1)

            # Apply Otsu's thresholding
            _, img_resized = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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


def prediction(model, inputs, outputs):
    predict = model.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    score = accuracy_score(outputs, predict)
    return score, margin_errors

def get_best_prams(inputs, outputs):
    hidden_layer_sizes_range = [(10, 50, 10), (20, 100, 20), (30, 150, 30)]
    max_iter_range = [30, 50, 100]
    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range, max_iter=max_iter_range)
    grid = GridSearchCV(MLPClassifier(alpha=0.005), param_grid, cv=10, scoring='accuracy')
    grid.fit(train_inputs, train_outputs)
    print(grid.cv_results_)
    print(grid.best_score_)
    print(grid.best_params_)


if __name__ == '__main__':
    if TRANSFORM_IMAGE:
        print("Transform image...")
        update_images(DATASET_TRAIN)
        update_images(DATASET_TEST)

    print("Getting data...")
    train_inputs, train_outputs = get_data(DATASET_TRAIN)
    get_best_prams(train_inputs, train_outputs)
    exit(0)


    if not exists(FILE_MODEL) or DO_LEARN:
        print("Learning...")
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER, max_iter=MAX_ITER, alpha=0.005)
        classifier.fit(train_inputs, train_outputs)
        save_model(classifier, FILE_MODEL)
    else:
        classifier = load_model(FILE_MODEL)

    print("Predicting...")
    test_inputs, test_outputs = get_data(DATASET_TEST)
    train_score, train_margin_errors = prediction(classifier, train_inputs, train_outputs)
    test_score, test_margin_errors = prediction(classifier, test_inputs, test_outputs)
    train_disp = ConfusionMatrixDisplay(confusion_matrix=train_margin_errors, display_labels=CLASSIFICATION)
    test_disp = ConfusionMatrixDisplay(confusion_matrix=test_margin_errors, display_labels=CLASSIFICATION)
    train_disp.plot()
    test_disp.plot()
    plt.show()
    print("Score training: " + str(train_score * 100) + "%\n")
    print("Score validation: " + str(test_score * 100) + "%\n")

