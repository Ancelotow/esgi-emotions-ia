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
TEMP_DIR = "temp"

# CLASSIFICATION
CLASSIFICATION = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Parameters
DO_LEARN = True
HIDDEN_LAYER = (50, 100, 50)
MAX_ITER = 30


def transform_and_get(directory):
    for i in range(len(CLASSIFICATION)):
        classification_folder = CLASSIFICATION[i]
        path = directory + "/" + classification_folder
        path_temp = path + "/" + TEMP_DIR
        filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        if not os.path.exists(path_temp):
            os.makedirs(path_temp)

        for fn in filenames:
            img_path = path + "/" + fn
            img_path_tmp = path_temp + "/" + fn
            img = io.imread(img_path, as_gray=True)  # Load the image in grayscale
            img_resized = resize(img, (64, 64))

            # Apply Sobel filter
            laplacian = cv2.Laplacian(np.float32(img_resized), cv2.CV_32F)
            laplacian = np.absolute(laplacian)
            max_value = np.max(laplacian)
            if max_value == 0:
                max_value = 1e-5  # small constant
            laplacian *= 255.0 / max_value

            cv2.imwrite(img_path_tmp, laplacian)


def get_data(directory):
    inputs = []
    outputs = []
    for i in range(len(CLASSIFICATION)):
        classification_folder = CLASSIFICATION[i]
        path = directory + "/" + classification_folder + "/" + TEMP_DIR
        filenames = os.listdir(path)
        for fn in filenames:
            img = io.imread(path + "/" + fn, as_gray=True)
            img_resized = resize(img, (64, 64))
            inputs.append(img_resized.flatten().tolist())
            outputs.append(i)
    return inputs, outputs


def prediction(model, inputs, outputs):
    predict = model.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    score = accuracy_score(outputs, predict)
    return score, margin_errors


if __name__ == '__main__':
    print("Transforming data...")
    transform_and_get(DATASET_TRAIN)
    transform_and_get(DATASET_TEST)
    print("Getting data...")
    train_inputs, train_outputs = get_data(DATASET_TRAIN)

    if not exists(FILE_MODEL) or DO_LEARN:
        print("Learning...")
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER, max_iter=MAX_ITER)
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

    train_score_formatted = "{:.2f}".format(train_score * 100)
    test_score_formatted = "{:.2f}".format(test_score * 100)
    print(f"\nScore training: {str(train_score_formatted)}%")
    print(f"Score validation: {str(test_score_formatted)}%")
