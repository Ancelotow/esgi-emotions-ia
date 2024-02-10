from matplotlib import pyplot as plt
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import os
from os.path import exists
from skimage import io
from api.lib.machine_learning import load_model, save_model

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
CLASSIFICATION = ["amber", "blue", "brown", "gray", "green", "hazel", "red"]

# Parameters
DO_LEARN = False
HIDDEN_LAYER_SIZE = (150, 250, 300, 250, 150)
MAX_ITER = 200
CURRENT_DATASET = DATASET_TEST


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
    print("Getting data...")
    inputs, outputs = get_data(CURRENT_DATASET)
    if not exists(FILE_MODEL) or DO_LEARN:
        # TRAIN
        print("Training...")
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
