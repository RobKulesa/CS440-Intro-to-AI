import numpy as np
import pandas as pd


class KNN:

    def __init__(self, df: pd.DataFrame, dset='digits'):
        self.data = df
        self.dset = dset

    def knn(self, test, k=500) -> int:
        distances = list()
        for i in range(self.data.shape[0]):
            distances.append((KNN.euclidean_distance(test, self.data.loc[i][:-1]),
                              self.data.loc[i]['label']))

        labels_sorted_by_distance = np.array(sorted(distances))[:, 1]
        labels_sorted_by_distance = labels_sorted_by_distance[:k]

        label_freq = np.unique(labels_sorted_by_distance, return_counts=True)

        freq_dict = dict()
        for i in range(len(label_freq[0])):
            freq_dict[label_freq[0][i]] = label_freq[1][i]
        prediction = int(max(freq_dict, key=freq_dict.get))
        return prediction

    @staticmethod
    def euclidean_distance(p1, p2) -> float:
        return np.power(np.sum(np.power(p2 - p1, 2)), 0.5)
