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
            # print(line)
            data.extend(translate_pixels(line))
        data = np.array(data, dtype=int)
        # print()
        if len(data) < width - 1:
            # we encountered end of file...
            print("Truncating at %d examples (maximum)" % i)
            break
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
    training_percent = float(input('Enter percent of training data to use (10, 20, etc): ')) / 100

    if dset == 'digits':
        width = 28
        height = 28
    else:
        width = 60
        height = 70

    # plt.imshow(data[0].reshape(height, width), cmap="gray")
    # plt.show()

    print('Getting training data...')
    training_data = load_data_file("data/digitdata/trainingimages", width, height)
    training_labels = load_labels_file("data/digitdata/traininglabels")
    df_train = pd.DataFrame(data=training_data)
    df_train.insert(loc=df_train.shape[1], column='label', value=training_labels)
    df_train_sample = df_train.sample(n=int(len(training_labels)*training_percent))

    print('Training KNN model...')
    my_knn = KNN(df_train_sample, dset='digits')

    print('Getting test data...')
    test_data = load_data_file("data/digitdata/testimages", width, height)
    test_labels = load_labels_file("data/digitdata/testlabels")

    k_vals = {35: 0, 21: 0, 15: 0, 11: 0, 5: 0, 3: 0, 1: 0}

    print('Testing %d%% of test data', ())
    for i, test in enumerate(test_data):
        predictions = my_knn.knn(test, k_vals)
        for idx, k_val in enumerate(k_vals):
            if predictions[idx] == test_labels[i]:
                k_vals[k_val] += 1

        if i > len(test_labels) * .95:
            print('\t95% complete...')
        elif i > test_size * .75:
            print('\t75% complete...')
        elif i > test_size * .5:
            print('\t50% complete...')
        elif i > test_size * .25:
            print('\t25% complete...')

    print(k_vals)


if __name__ == '__main__':
    main()
