import os

from PIL import Image

from api.main import DATASET_FILE, CLASSIFICATION

from api.model.model import Model


def write_header_dataset():
    line = []
    for i in range(1, 49):
        for j in range(1, 49):
            h = "px" + str(i * j)
            line.append(h)
    line.append("emotion")
    with open(DATASET_FILE, "w") as fd:
        fd.write(','.join(line))
        fd.write("\n")


def write_dataset_file(dir):
    with open(DATASET_FILE, "a") as fd:
        for i in range(len(CLASSIFICATION)):
            emotion = CLASSIFICATION[i]
            path = dir + "/" + emotion
            filenames = os.listdir(path)
            for fn in filenames:
                with Image.open(path + "/" + fn) as img:
                    em = Model(emotion, img)
                    csv = [str(value) for value in em.to_array()]
                    csv.append(str(i))
                    fd.write(','.join(csv) + "\n")