import os
import cv2
import numpy as np
from docutils.nodes import Sequential
from skimage.transform import resize
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from os.path import exists
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


PATH = "feelings"
# DATASETS
DIR_MODEL = "../../dataset/"+PATH+"/model_directory"
FILE_MODEL = "../../dataset/"+PATH+"/model.dat"
DATASET_TRAIN = "../../dataset/"+PATH+"/train"
DATASET_TEST = "../../dataset/"+PATH+"/test"
TEMP_DIR = "temp"

# CLASSIFICATION
CLASSIFICATION = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Parameters
DO_LEARN = False
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
    predict_labels = np.argmax(predict, axis=-1)
    margin_errors = confusion_matrix(outputs, predict_labels, normalize='true')
    score = accuracy_score(outputs, predict_labels)
    return score, margin_errors


if __name__ == '__main__':
    print("Transforming data...")
    transform_and_get(DATASET_TRAIN)
    transform_and_get(DATASET_TEST)
    print("Getting data...")
    train_inputs, train_outputs = get_data(DATASET_TRAIN)

    if not exists(FILE_MODEL) or DO_LEARN:
        print("Learning...")
        np_inputs = np.array(train_inputs).reshape(-1, 64, 64, 1)
        # Define the CNN model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(CLASSIFICATION), activation='softmax'))  # Output layer
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Train the model
        np_outputs = np.array(train_outputs)
        print("Fitting...")
        model.fit(np_inputs, np_outputs, epochs=MAX_ITER)
        # Save the model
        model.save(DIR_MODEL)
    else:
        model = load_model(DIR_MODEL)

    print("Predicting...")
    train_inputs = np.array(train_inputs).reshape(-1, 64, 64, 1)
    train_score, train_margin_errors = prediction(model, train_inputs, train_outputs)

    test_inputs, test_outputs = get_data(DATASET_TEST)
    test_inputs = np.array(test_inputs).reshape(-1, 64, 64, 1)
    test_score, test_margin_errors = prediction(model, test_inputs, test_outputs)

    train_disp = ConfusionMatrixDisplay(confusion_matrix=train_margin_errors, display_labels=CLASSIFICATION)
    test_disp = ConfusionMatrixDisplay(confusion_matrix=test_margin_errors, display_labels=CLASSIFICATION)
    train_disp.plot()
    test_disp.plot()
    plt.show()

    train_score_formatted = "{:.2f}".format(train_score * 100)
    test_score_formatted = "{:.2f}".format(test_score * 100)
    print(f"\nScore training: {str(train_score_formatted)}%")
    print(f"Score validation: {str(test_score_formatted)}%")
