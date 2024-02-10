import os
import cv2
import numpy as np
from keras import Model, Input
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.regularizers import l2
from skimage.transform import resize
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import confusion_matrix
from os.path import exists
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

PATH = "feelings"
# DATASETS
DIR_MODEL = "../../dataset/"+PATH+"/model_directory"
FILE_MODEL = "../../dataset/"+PATH+"/model.h5"
DATASET_TRAIN = "../../dataset/"+PATH+"/train"
DATASET_TEST = "../../dataset/"+PATH+"/test"
TEMP_DIR = "temp"

# CLASSIFICATION
CLASSIFICATION = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Parameters
DO_LEARN = True
MAX_ITER = 35


def transform(directory):
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

            # Increase contrast
            img_resized = cv2.equalizeHist((img_resized * 255).astype(np.uint8))

            # Apply Sobel filter
            sobelx = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 1, 0, ksize=9)
            sobely = cv2.Sobel(np.float32(img_resized), cv2.CV_64F, 0, 1, ksize=9)
            sobel = np.hypot(sobelx, sobely)
            max_value = np.max(sobel)
            if max_value == 0:
                max_value = 1e-5  # small constant
            sobel *= 255.0 / max_value


            cv2.imwrite(img_path_tmp, sobel)


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


def prediction(classifier_model, inputs, outputs):
    predict = classifier_model.predict(inputs)
    predict_labels = np.argmax(predict, axis=-1)
    margin_errors = confusion_matrix(outputs, predict_labels, normalize='true')
    score = accuracy_score(outputs, predict_labels)
    return score, margin_errors


if __name__ == '__main__':
    print("Transforming data...")
    transform(DATASET_TRAIN)
    transform(DATASET_TEST)
    print("Getting data...")
    train_inputs, train_outputs = get_data(DATASET_TRAIN)
    test_inputs, test_outputs = get_data(DATASET_TEST)


    if not exists(FILE_MODEL) or DO_LEARN:
        print("Modeling...")
        # Define the CNN model
        input = Input(shape=(64, 64, 1))
        conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
        conv1 = Dropout(0.1)(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
        conv2 = Dropout(0.1)(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flatten = Flatten()(pool2)
        dense_1 = Dense(128, activation='relu')(flatten)
        drop_1 = Dropout(0.2)(dense_1)
        output = Dense(len(CLASSIFICATION), activation="softmax")(drop_1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        # Compile the model
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        model.summary()
        # Train the model
        print("Fitting...")
        np_outputs = np.array(train_outputs)
        np_inputs = np.array(train_inputs).reshape(-1, 64, 64, 1)
        np_test_inputs = np.array(test_inputs).reshape(-1, 64, 64, 1)
        np_test_outputs = np.array(test_outputs)
        classifier = model.fit(np_inputs, np_outputs, batch_size=32, validation_data=(np_test_inputs, np_test_outputs), epochs=MAX_ITER, callbacks=[early_stopping])
        model.save(FILE_MODEL)

        train_loss = classifier.history['loss']
        test_loss = classifier.history['val_loss']
        train_accuracy = classifier.history['accuracy']
        test_accuracy = classifier.history['val_accuracy']

        # Plotting a line chart to visualize the loss and accuracy values by epochs.
        fig1, ax1 = plt.subplots(figsize=(15, 7))
        ax1.plot(train_loss, label='Train Loss', color='royalblue', marker='o', markersize=5)
        ax1.plot(test_loss, label='Test Loss', color='orangered', marker='o', markersize=5)
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Categorical Crossentropy', fontsize=14)
        ax1.legend(fontsize=14)
        ax1.tick_params(axis='both', labelsize=12)
        fig1.suptitle("Loss of CNN Models", fontsize=16)
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(15, 7))
        ax2.plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
        ax2.plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        ax2.legend(fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        fig2.suptitle("Accuracy of CNN Models", fontsize=16)
        plt.show()
        # Save the model
    else:
        model = load_model(FILE_MODEL)

    print("Predicting...")
    train_inputs = np.array(train_inputs).reshape(-1, 64, 64, 1)
    train_score, train_margin_errors = prediction(model, train_inputs, train_outputs)

    test_inputs = np.array(test_inputs).reshape(-1, 64, 64, 1)
    test_score, test_margin_errors = prediction(model, test_inputs, test_outputs)

    plt.figure(figsize=(10, 10))
    train_disp = ConfusionMatrixDisplay(confusion_matrix=train_margin_errors, display_labels=CLASSIFICATION)
    train_disp.plot()
    plt.title('Train Confusion Matrix')
    plt.show()

    plt.figure(figsize=(10, 10))
    test_disp = ConfusionMatrixDisplay(confusion_matrix=test_margin_errors, display_labels=CLASSIFICATION)
    test_disp.plot()
    plt.title('Test Confusion Matrix')
    plt.show()

    train_score_formatted = "{:.2f}".format(train_score * 100)
    test_score_formatted = "{:.2f}".format(test_score * 100)
    print(f"\nScore training: {str(train_score_formatted)}%")
    print(f"Score validation: {str(test_score_formatted)}%")

