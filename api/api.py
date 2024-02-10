import cv2
import numpy as np
from api.lib.executor.feelings import FEELINGS_CLASS
from tensorflow.keras.models import load_model
from api.lib.transformer import transform_for_feeling

feelings_model = load_model("../api/dataset/feelings/model.h5")


def feeling_detection(image):
    img_transformed = transform_for_feeling(image)
    cv2.imwrite("test.png", img_transformed)
    img_reshaped = img_transformed.reshape(-1, 64, 64, 1)  # reshape the flattened image
    predict = feelings_model.predict(img_reshaped)
    predict_feeling = FEELINGS_CLASS[np.argmax(predict[0])]
    return predict_feeling