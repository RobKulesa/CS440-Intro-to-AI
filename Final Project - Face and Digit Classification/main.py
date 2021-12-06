from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from knn import KNN


def load_labels_file(filename: str, n: int):
    lines = read_lines(filename)
    labels = list()
    for line in lines[:min(n, len(lines))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def load_data_file(filename: str, n: int, width: int, height: int) -> np.array:
    lines = read_lines(filename)
    lines.reverse()
    items = list()
    for i in range(n):
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
    # plt.imshow(data[0].reshape(height, width), cmap="gray")
    # plt.show()
    width = 28
    height = 28

    training_size = 5000
    training_data = load_data_file("data/digitdata/trainingimages", training_size, width, height)
    training_labels = load_labels_file("data/digitdata/traininglabels", training_size)
    df_train = pd.DataFrame(data=training_data)
    df_train.insert(loc=df_train.shape[1], column='label', value=training_labels)
    my_knn = KNN(df_train)

    test_size = 1000
    test_data = load_data_file("data/digitdata/testimages", test_size, width, height)
    test_labels = load_labels_file("data/digitdata/testlabels", test_size)

    i = 66
    prediction = my_knn.knn(test_data[i], k=500)
    print(f'Prediction is {prediction}, actual is {test_labels[i]}')


if __name__ == '__main__':
    main()
