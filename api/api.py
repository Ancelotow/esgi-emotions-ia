import numpy as np

from api.lib.machine_learning import load_model

CLASSIFICATION = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
classifier = load_model("../api/dataset/feelings/model.dat")

def feeling_detection(image):
    predict = classifier.predict(image)
    predict_feeling = CLASSIFICATION[int(predict[0])]
    return predict_feeling
