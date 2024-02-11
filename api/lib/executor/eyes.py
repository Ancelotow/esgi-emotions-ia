import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
import os
from os.path import exists
from skimage import io
from sklearn.model_selection import train_test_split
from api.lib.machine_learning import load_model, save_model
from api.lib.preprocessing import preprocess_for_eyes

PATH = "eyes"
FILE_MODEL = "../../dataset/" + PATH + "/model.dat"
DATASET = "../../dataset/" + PATH + "/train"
TEMP_DIR = "temp"

# CLASSIFICATION
EYE_CLASS = ["amber", "blue", "brown", "gray", "green", "hazel", "red"]

# Parameters
DO_LEARN = True
HIDDEN_LAYER_SIZE = (150, 250, 300, 250, 150)
MAX_ITER = 200


def transform(directory):
    for i in range(len(EYE_CLASS)):
        classification_folder = EYE_CLASS[i]
        path = directory + "/" + classification_folder
        path_temp = path + "/" + TEMP_DIR
        filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        if not os.path.exists(path_temp):
            os.makedirs(path_temp)

        for fn in filenames:
            img_path = path + "/" + fn
            img_path_tmp = path_temp + "/" + fn
            img = io.imread(img_path)
            img_transformed = preprocess_for_eyes(img)
            cv2.imwrite(img_path_tmp, img_transformed)


def get_data(directory):
    inputs = []
    outputs = []
    for i in range(len(EYE_CLASS)):
        classification_folder = EYE_CLASS[i]
        path = directory + "/" + classification_folder + "/" + TEMP_DIR
        filenames = os.listdir(path)
        for fn in filenames:
            img = io.imread(path + "/" + fn)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs.append(img_bgr.flatten().tolist())
            outputs.append(i)
    return inputs, outputs


def prediction(model, inputs, outputs):
    predict = model.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    score = accuracy_score(outputs, predict)
    return score, margin_errors


if __name__ == '__main__':
    print("Transforming...")
    transform(DATASET)
    print("Getting data...")
    all_inputs, all_outputs = get_data(DATASET)
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(all_inputs, all_outputs, test_size=0.6)
    if not exists(FILE_MODEL) or DO_LEARN:
        print("Training...")
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=MAX_ITER)
        classifier.fit(train_inputs, train_outputs)
        save_model(classifier, FILE_MODEL)
    else:
        classifier = load_model(FILE_MODEL)

    print("Predicting...")
    train_score, train_errors = prediction(classifier, train_inputs, train_outputs)
    test_score, test_errors = prediction(classifier, test_inputs, test_outputs)

    plt.figure(figsize=(10, 10))
    train_disp = ConfusionMatrixDisplay(confusion_matrix=train_errors, display_labels=EYE_CLASS)
    train_disp.plot()
    plt.title('Train Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 10))
    test_disp = ConfusionMatrixDisplay(confusion_matrix=test_errors, display_labels=EYE_CLASS)
    test_disp.plot()
    plt.title('Test Confusion Matrix')
    plt.show()

    train_score_formatted = "{:.2f}".format(train_score * 100)
    test_score_formatted = "{:.2f}".format(test_score * 100)
    print(f"\nScore training: {str(train_score_formatted)}%")
    print(f"Score validation: {str(test_score_formatted)}%")
