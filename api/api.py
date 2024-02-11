import pickle

import cv2
import numpy as np

from api.lib.executor.eyes import EYE_CLASS
from api.lib.executor.feelings import FEELINGS_CLASS
from tensorflow.keras.models import load_model
from api.lib.preprocessing import preprocess_for_feeling, preprocess_for_age, preprocess_for_eyes


def load_dat_model(file_model):
    with open(file_model, 'rb') as file:
        classifier = pickle.load(file)
    return classifier


feelings_model = load_model("../api/dataset/feelings/model.h5")
age_model = load_dat_model("../api/dataset/age/model.dat")
eye_model = load_dat_model("../api/dataset/eyes/model.dat")


def feeling_detection(image):
    img_transformed = preprocess_for_feeling(image)
    cv2.imwrite("feeling.png", img_transformed)
    img_reshaped = img_transformed.reshape(-1, 64, 64, 1)  # reshape the flattened image
    predict = feelings_model.predict([img_reshaped])
    predict_feeling = FEELINGS_CLASS[np.argmax(predict[0])]
    return predict_feeling


def age_detection(image):
    img_transformed = preprocess_for_age(image)
    intput = img_transformed.flatten().tolist()
    cv2.imwrite("year.png", img_transformed)
    predict = age_model.predict([intput])
    predict_age = predict[0]
    return int(predict_age)


def eye_detection(image):
    img_clr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_transformed = preprocess_for_eyes(img_clr)
    intput = img_transformed.flatten().tolist()
    cv2.imwrite("eyes.png", img_transformed)
    predict = eye_model.predict([intput])
    predict_eye = EYE_CLASS[int(predict[0])]
    return predict_eye
