from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from knn import KNN


def load_labels_file(filename: str):
    lines = read_lines(filename)
    labels = list()
    for line in lines:
        if line == '':
            break
        labels.append(int(line))
    return labels


def load_data_file(filename: str, width: int, height: int) -> np.array:
    lines = read_lines(filename)
    lines.reverse()
    items = list()
    while len(lines) > 0:
        data = list()
        for j in range(height):
            if len(lines) == 0:
                return np.array(items)
            line = lines.pop()
            data.extend(translate_pixels(line))
        data = np.array(data, dtype=int)
        items.append(data)
    return np.array(items)


def read_lines(filename: str):
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        raise ValueError("Invalid data filename")


def translate_pixels(line: str) -> List[int]:
    new_line = list()
    for c in line:
        if c == '+':
            new_line.append(1)
        elif c == '#':
            new_line.append(2)
        else:
            new_line.append(0)
    return new_line


def main():
    dset = input('Choose dataset (face, digits): ')
    training_percent = int(input('Enter percent of training data to use (10, 20, etc): '))

    print('Getting data...')

    if dset == 'digits':
        width = 28
        height = 28
        training_data = load_data_file("data/digitdata/trainingimages", width, height)
        training_labels = load_labels_file("data/digitdata/traininglabels")
        test_data = load_data_file("data/digitdata/testimages", width, height)
        test_labels = load_labels_file("data/digitdata/testlabels")
    else:
        width = 60
        height = 70
        training_data = load_data_file("data/facedata/facedatatrain", width, height)
        training_labels = load_labels_file("data/facedata/facedatatrainlabels")
        test_data = load_data_file("data/facedata/facedatatest", width, height)
        test_labels = load_labels_file("data/facedata/facedatatestlabels")

    df_train = pd.DataFrame(data=training_data)
    df_train.insert(loc=df_train.shape[1], column='label', value=training_labels)
    if training_percent == 100:
        df_train_sample = df_train
    else:
        df_train_sample = df_train.sample(frac=training_percent / 100)

    print('Training KNN model...')
    my_knn = KNN(df_train_sample, dset=dset)

    k_vals = {35: 0, 21: 0, 15: 0, 11: 0, 5: 0, 3: 0, 1: 0}

    print_counter = 0
    print('Testing on %d%% of training data' % training_percent)
    for i, test in enumerate(test_data):
        predictions = my_knn.knn(test, k_vals)
        for idx, k_val in enumerate(k_vals):
            if predictions[idx] == test_labels[i]:
                k_vals[k_val] += 1

        if i > len(test_labels) * .95 and print_counter == 3:
            print_counter += 1
            print('\t95% complete...')
        elif i > len(test_labels) * .75 and print_counter == 2:
            print_counter += 1
            print('\t75% complete...')
        elif i > len(test_labels) * .5 and print_counter == 1:
            print_counter += 1
            print('\t50% complete...')
        elif i > len(test_labels) * .25 and print_counter == 0:
            print('\t25% complete...')
            print_counter += 1

    print(k_vals)


if __name__ == '__main__':
    main()
