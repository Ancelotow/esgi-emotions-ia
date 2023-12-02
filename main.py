import os
from PIL import Image
import numpy
from emotion_data import EmotionData

DATASET_TRAIN = "./dataset/train"
DATASET_TEST = "./dataset/test"
LEARNING_RATE = 1E-3
EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
DATASET_FILE = "dataset.csv"


def write_header_dataset():
    line = []
    for i in range(1, 49):
        for j in range(1, 49):
            h = "px" + str(i * j)
            line.append(h)
    line.append("emotion")
    with open(DATASET_FILE, "w") as fd:
        fd.write(','.join(line))


def write_dataset_file(dir):
    with open(DATASET_FILE, "a") as fd:
        for i in range(len(EMOTIONS)):
            emotion = EMOTIONS[i]
            print("==========================", emotion, "==========================")
            path = dir + "/" + emotion
            filenames = os.listdir(path)
            for fn in filenames:
                print(fn)
                with Image.open(path + "/" + fn) as img:
                    em = EmotionData(emotion, img)
                    csv = [str(value) for value in em.to_array()]
                    csv.append(str(i))
                    fd.write(','.join(csv))


if __name__=="__main__":
    os.remove(DATASET_FILE)
    write_header_dataset()
    write_dataset_file(DATASET_TRAIN)