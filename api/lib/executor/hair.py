from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import os
from os.path import exists
from PIL import Image
import platform
import cv2

from api.lib.dataset import write_header_dataset, write_dataset_file, rescale_image
from api.lib.machine_learning import get_data, load_model, save_model

PATH = "hair"
# DATASETS
FILE_MODEL = "../../dataset/" + PATH + "/model.dat"
DATASET_TRAIN = "../../dataset/" + PATH + "/train"
DATASET_TEST = "../../dataset/" + PATH + "/test"
DATASET_FILE = "../../dataset/" + PATH + "/dataset_train.csv"

DATASET_NAME_TEST = "../../dataset/" + PATH + "/dataset_train.csv"
DATASET_NAME_TRAIN = "../../dataset/" + PATH + "/dataset_train.csv"
DATASET_NAME_PREDICT = "../../dataset/" + PATH + "/dataset_predict.csv"

# CLASSIFICATION
CLASSIFICATION = ["curly", "straight", "wavy"]

# LEARNING
DO_LEARN = False
HIDDEN_LAYER_SIZE = (300, 300)

if __name__ == '__main__':
    # os.remove(DATASET_FILE)
    # Creation dataset.csv
    rescale_image(DATASET_TRAIN, 100)
    write_header_dataset("eyes", 100, DATASET_FILE)
    write_dataset_file(DATASET_TEST, CLASSIFICATION, DATASET_FILE)

    inputs, outputs = get_data("TEST", PATH)
    if not exists(FILE_MODEL) or DO_LEARN:
        # TRAIN
        classifier = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=2000)
        classifier.fit(inputs, outputs)
        margin_errors = classifier.score(inputs, outputs)
        print("Margin of errors ", str(margin_errors))
        # plt.plot(classifier.loss_curve_, color="blue")
        # plt.show()
        save_model(classifier, PATH)
    else:
        classifier = load_model(FILE_MODEL)

    predict = classifier.predict(inputs)
    margin_errors = confusion_matrix(outputs, predict, normalize='true')
    print("Errors : \n" + str(margin_errors))

    # PREDICT
    inputs = []
    with Image.open("../../dataset/predict/face.png") as img:
        for x in range(img.width):
            for y in range(img.height):
                inputs.append(img.getpixel((x, y)))
    predict = classifier.predict([inputs])
    predict_hair = CLASSIFICATION[int(predict[0])]
    print("HAIR Predict : " + str(predict_hair))
