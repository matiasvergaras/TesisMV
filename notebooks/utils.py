import os
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from textwrap import wrap


class KunischMetrics:
    def __init__(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        self.y_true = y_true
        self.y_pred = y_pred

    def get_f1(self, average='macro'):
        return f1_score(self.y_true, self.y_pred, average=average)

    def get_recall(self, average='macro'):
        return recall_score(self.y_true, self.y_pred, average=average)

    def get_precision(self, average='macro'):
        return precision_score(self.y_true, self.y_pred, average=average)

    def get_accuracy(self, normalize=True):
        return accuracy_score(self.y_true, self.y_pred, normalize=normalize)

    def get_hamming_loss(self, normalize=True):
        # Compute the average Hamming loss.
        # The Hamming loss is the fraction of labels that are incorrectly predicted.
        return hamming_loss(self.y_true, self.y_pred)

    def get_emr(self):
        # Compute Exact Match Ratio.
        return np.all(self.y_pred == self.y_true, axis=1).mean()

    def get_hamming_score(self):
        # Compute Hamming Score
        # Hamming Score = |Intersección de positivos|/|Unión de positivos|, promediado por la cantidad de samples
        # También se puede ver como la proporción de etiquetas correctamente asignadas sobre la cantidad total de
        # etiquetas asignadas. Se conoce además como Multilabel Accuracy, y "castiga" por: (1) no predecir una etiqueta
        # correcta (disminuyendo la cardinalidad de la intersección) y (2) incluir una etiqueta incorrecta (aumentando
        # la cardinalidad de la unión).
        temp = 0
        for i in range(self.y_true.shape[0]):
            temp += sum(np.logical_and(self.y_true[i], self.y_pred[i])) / sum(
                np.logical_or(self.y_true[i], self.y_pred[i]))
        return temp / self.y_true.shape[0]

    # Nos gustaria tener una metrica que mida al menos cuantas etiquetas son predichas correctamente para cada patron
    # Y quizá sería interesante relacionar eso con el label cardinality (que en nuestro dataset es 5.28)
    def match_ratio_at(self, n=5):
        count = 0
        for i in range(self.y_true.shape[0]):
            count += (1 if sum(np.logical_and(self.y_true[i], self.y_pred[i])) >= n else 0)
        return count / self.y_true.shape[0]


class KunischPrunner:
    def __init__(self, desired_labels, pruning_freq=26):
        self.desired_labels = desired_labels
        self.final_labels = -1
        self.pruning_freq = 26
        self.top_labels = None

    def filter_labels(self, labels_df, desired_labels, pruning_freq=None):
        top_labels = None

        if pruning_freq is not None:
            print(f"Utilizando pruning frequency.\
            Se ignorará la cantidad deseada de etiquetas para cortar en {pruning_freq} eventos.")
            filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pruning_freq]
            top_labels = filtered_df.sum().sort_values(ascending=False)
            return top_labels

        else:
            filtered_labels = labels_df.shape[1]
            pivot = 0
            while filtered_labels > desired_labels:
                filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pivot]
                top_labels = filtered_df.sum().sort_values(ascending=False)
                filtered_labels = filtered_df.shape[1]
                pivot += 1
            print("Aplicando threshold {} para trabajar con {} labels".format(pivot, len(top_labels.values)))
            self.final_labels = len(top_labels.values)
            self.top_labels = top_labels.values
            return top_labels

    def filter_df(self, df):
        df = df[df.columns.intersection(self.top_labels.index)]
        return df

    def combine_dfs(self, labels_df, features_df):
        assert len(labels_df) == len(features_df)
        labels_df = labels_df[labels_df.columns.intersection(self.top_labels.index)]
        final_df = pd.merge(labels_df,
                            features_df,
                            left_index=True, right_index=True)
        return final_df
