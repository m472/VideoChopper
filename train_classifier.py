#!/usr/bin/python3

import os
import random
from glob import glob

import numpy as np
from mypy.types import List
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class TrainingData:
    def __init__(self, data: List[np.ndarray], labels: List[int]):
        self.data: List[np.ndarray] = data
        self.labels: List[int] = labels


def create_model(training_data: TrainingData) -> Sequential:
    model = Sequential()

    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(20, 3)))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(60, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    x = np.stack(training_data.data)
    y = to_categorical(np.array(training_data.labels))
    n_training = x.shape[0] // 2

    x_train = x[:n_training]
    y_train = y[:n_training]
    x_valid = x[n_training:]
    y_valid = y[n_training:]

    model.fit(x_train, y_train, epochs=100, batch_size=100, validation_data=(x_valid, y_valid))

    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_valid, y_valid, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    return model


def load_training_data(filename_filters: List[str], labels: List[int]) -> TrainingData:
    indices, data = [], []
    for filename_filter, label in zip(filename_filters, labels):
        for data_name in glob(filename_filter):
            loaded = np.uint8(np.load(data_name, allow_pickle=False))
            data.append(loaded)
            indices.append(label)

    # shuffle data
    combined = list(zip(indices, data))
    random.shuffle(combined)
    indices, data = list(zip(*combined))

    return TrainingData(data, indices)


def preprocess(data: TrainingData) -> TrainingData:
    return data


if __name__ == "__main__":
    import pickle

    directory = '/run/media/matz/SD 32GB/VideoChopper/'
    training_files = [('out_1_*.npy', 'Matz'),
                      ('out_2_*.npy', 'Flavia'),
                      ('out_3_*.npy', 'Flavia'),
                      ('out_4_*.npy', 'Matz'),
                      ('out_5_*.npy', 'Janine')]
    filename_patterns, labels = zip(*training_files)
    names = set(labels)
    name_lookup = {index: value for index, value in enumerate(names)}
    reverse_name_lookup = {value: index for index, value in enumerate(names)}
    data = preprocess(load_training_data([os.path.join(directory, f) for f in filename_patterns],
                                         [reverse_name_lookup[name] for name in labels]))

    with open('names', 'wb') as f:
        pickle.dump((name_lookup, reverse_name_lookup), f)

    model = create_model(data)
    model.save('model')
