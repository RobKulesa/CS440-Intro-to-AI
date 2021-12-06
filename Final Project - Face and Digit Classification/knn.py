from typing import List, Union

import numpy as np
import pandas as pd


class KNN:

    def __init__(self, df: pd.DataFrame, dset='digits'):
        self.data = df
        self.dset = dset

    def knn(self, test, k_vals: dict) -> List[int]:
        distances = list()
        for i in range(self.data.shape[0]):
            if self.dset == 'digits':
                distances.append((KNN.euclidean_distance(test, self.data.loc[i][:-1]),
                                  self.data.loc[i]['label']))
            else:
                distances.append((KNN.cosine_distance(test, self.data.loc[i][:-1]),
                                 self.data.loc[i]['label']))

        labels_sorted_by_distance = np.array(sorted(distances))[:, 1]
        predictions = list()
        for k_val in k_vals:
            labels_sorted_by_distance = labels_sorted_by_distance[:k_val]

            label_freq = np.unique(labels_sorted_by_distance, return_counts=True)

            freq_dict = dict()
            for i in range(len(label_freq[0])):
                freq_dict[label_freq[0][i]] = label_freq[1][i]
            predictions.append(int(max(freq_dict, key=freq_dict.get)))

        return predictions

    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        return np.sqrt(np.sum(np.power(p2 - p1, 2)))

    @staticmethod
    def cosine_distance(p1, p2) -> float:
        return 1 - (np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))
