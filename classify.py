#!/usr/bin/python3
from tensorflow import keras
from glob import glob
import numpy as np


def load_data(filename_filter: str):
    data = []
    for data_name in glob(filename_filter):
        loaded = np.uint8(np.load(data_name, allow_pickle=False))
        data.append(loaded)

    return np.stack(data)


def classify(model, data) -> (int, float):
    average_classification = np.average(np.stack(model.predict(data)), axis=0)
    result = np.argmax(average_classification)
    certainty = average_classification[result]
    return int(result), certainty


if __name__ == '__main__':
    import pickle

    model = keras.models.load_model('model')
    with open('names', 'rb') as f:
        name_lookup, reverse_name_lookup = pickle.load(f)

    for i in range(12):
        data = load_data(f'/run/media/matz/SD 32GB/VideoChopper/out_{i + 1}_*.npy')
        label, certainty = classify(model, data)
        name = name_lookup[label]
        print(f'{name} ({certainty * 100:.1f}%)')
