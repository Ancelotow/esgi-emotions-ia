import os

import cv2
import matplotlib.pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from api.lib.machine_learning import save_model, load_model
from api.lib.preprocessing import preprocess_for_age

PATH = "age"
DATASET = "../../dataset/" + PATH + "/test"
FILE_MODEL = "../../dataset/" + PATH + "/model.dat"
TEMP_DIR = "temp"

# Parameters
DO_LEARN = True
MAX_ITER = 100
HIDDEN_LAYER_SIZE = (50, 100, 150, 100, 50)


def transform():
    path_temp = DATASET + "/" + TEMP_DIR
    filenames = [f for f in os.listdir(DATASET) if os.path.isfile(os.path.join(DATASET, f))]

    if not os.path.exists(path_temp):
        os.makedirs(path_temp)

    for f in filenames:
        filename = DATASET + "/" + f
        filename_tmp = path_temp + "/" + f
        img = io.imread(filename)
        img_transformed = preprocess_for_age(img)
        cv2.imwrite(filename_tmp, img_transformed)


def get_data():
    inputs = []
    outputs = []
    files = os.listdir(DATASET + "/" + TEMP_DIR)
    for f in files:
        filename = DATASET + "/" + TEMP_DIR + "/" + f
        img = io.imread(filename)
        inputs.append(img.flatten().tolist())
        infos = f.split('_')
        outputs.append(int(infos[0]))
    return inputs, outputs


if __name__ == '__main__':
    print("Transforming...")
    transform()
    print("Getting data...")
    all_inputs, all_outputs = get_data()
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(all_inputs, all_outputs, test_size=0.6)
    if DO_LEARN:
        print("Learning...")
        regressor = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=MAX_ITER)
        regressor.fit(train_inputs, train_outputs)
        save_model(regressor, FILE_MODEL)
    else:
        regressor = load_model(FILE_MODEL)

    print("Validation...")
    result = regressor.score(test_inputs, test_outputs)
    plt.plot(regressor.loss_curve_, color="orange")
    print(result)
    predicated = regressor.predict(test_inputs)
    errors = []
    for i in range(len(test_inputs)):
        errors.append(abs(predicated[i] - test_outputs[i]))
    avg_error = sum(errors) / len(errors)
    min_error = min(errors)
    max_error = max(errors)
    print(f'error min: {min_error} - error avg: {avg_error} - error max: {max_error}')
    plt.show()



