import os

from PIL import Image
from api.model.model import Model


def write_header_dataset(classification: str, size_image: int, file: str):
    line = []
    for i in range(1, size_image):
        for j in range(1, size_image):
            h = "px" + str(i * j)
            line.append(h)
    line.append(classification)
    with open(file, "w") as fd:
        fd.write(','.join(line))
        fd.write("\n")


def write_dataset_file(dir, classification_object: [any], dataset_file, ):
    with open(dataset_file, "a") as fd:
        for i in range(len(classification_object)):
            classification_folder = classification_object[i]
            path = dir + "/" + classification_folder
            filenames = os.listdir(path)
            for fn in filenames:
                with Image.open(path + "/" + fn) as img:
                    em = Model(classification_folder, img)
                    csv = [str(value) for value in em.to_array()]
                    csv.append(str(i))
                    fd.write(','.join(csv) + "\n")


def rescale_image(dir, size_image: int):
    for folder in os.listdir(dir):
        path = dir + "/" + folder
        filenames = os.listdir(path)
        for fn in filenames:
            with Image.open(path + "/" + fn) as img:
                try:
                    img = img.convert('L')
                    img = img.resize((size_image, size_image))
                    img.save(path + "/" + fn)
                except Exception as e:
                    print(e)
