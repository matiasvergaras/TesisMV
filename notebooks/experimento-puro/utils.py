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
from scipy import sparse
import random


class KunischMetrics:
    def __init__(self, y_true, y_pred):
        assert y_true.shape == y_pred.shape
        if sparse.issparse(y_true):
            y_true = y_true.todense()
        if sparse.issparse(y_pred):
            y_pred = y_pred.todense()
        self.y_true = y_true
        self.y_pred = y_pred

    def f1(self, average='micro'):
        return round(f1_score(self.y_true, self.y_pred, average=average), 4)

    def recall(self, average='micro'):
        return round(recall_score(self.y_true, self.y_pred, average=average), 4)

    def precision(self, average='micro'):
        return round(precision_score(self.y_true, self.y_pred, average=average), 4)

    def acc(self, normalize=True):
        return round(accuracy_score(self.y_true, self.y_pred, normalize=normalize), 4)

    def hl(self, normalize=True):
        # Compute the average Hamming loss.
        # The Hamming loss is the fraction of labels that are incorrectly predicted.
        return round(hamming_loss(self.y_true, self.y_pred), 4)

    def emr(self):
        # Compute Exact Match Ratio.
        return round(np.all(self.y_pred == self.y_true, axis=1).mean(), 4)

        # Compute Hamming Score
        # Hamming Score = |Intersección de positivos|/|Unión de positivos|, promediado por la cantidad de samples
        # También se puede ver como la proporción de etiquetas correctamente asignadas sobre la cantidad total de
        # etiquetas asignadas. Se conoce además como Multilabel Accuracy, y "castiga" por: (1) no predecir una etiqueta
        # correcta (disminuyendo la cardinalidad de la intersección) y (2) incluir una etiqueta incorrecta (aumentando
        # la cardinalidad de la unión).

    def hs(self):
        acc_list = []
        for i in range(self.y_true.shape[0]):
            set_true = set(np.where(self.y_true[i])[0])
            set_pred = set(np.where(self.y_pred[i])[0])
            # print('\nset_true: {0}'.format(set_true))
            # print('set_pred: {0}'.format(set_pred))
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / \
                        float(len(set_true.union(set_pred)))
            # print('tmp_a: {0}'.format(tmp_a))
            acc_list.append(tmp_a)
        return round(np.mean(acc_list), 4)

    # Nos gustaria tener una metrica que mida al menos cuantas etiquetas son predichas correctamente para cada patron
    # Y quizá sería interesante relacionar eso con el label cardinality (que en nuestro dataset es 5.28)
    def k_match_ratio(self, n=5, strict=False):
        count = 0
        for i in range(self.y_true.shape[0]):
            tp = np.logical_and(self.y_true[i], self.y_pred[i])
            tp = np.sum(tp)
            if strict:
                if (tp >= n) or (tp == self.y_true.shape[0]):
                    count += 1
            else:
                count += 1 if tp >= n else 0
        return round(count / self.y_true.shape[0], 4)

    def mr1(self):
        return self.k_match_ratio(1)

    def mr2(self):
        return self.k_match_ratio(2)

    def mr3(self):
        return self.k_match_ratio(3)

    def mr4(self):
        return self.k_match_ratio(4)

    def mr5(self):
        return self.k_match_ratio(5)


class KunischPruner:
    def __init__(self, desired_labels, pruning_freq=26):
        self.desired_labels = desired_labels
        self.final_labels = -1
        self.pruning_freq = 26
        self.top_labels = None

    def set_top_labels(self, top_labels):
        self.top_labels = top_labels

    def filter_labels(self, labels_df, pruning_freq=None):
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
            while filtered_labels > self.desired_labels:
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


class DataExplorer:
    def __init__(self, train_labels, val_labels, test_labels):
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

    def get_unique_combinations(self, study='train'):
        if study == 'train':
            labels_df = self.train_labels
        elif study == 'val':
            labels_df = self.val_labels
        elif study == 'test':
            labels_df = self.test_labels
        elif study == 'all':
            labels_df = pd.concat([self.train_labels, self.val_labels, self.test_labels])

        unique_combinations = len(labels_df.drop_duplicates())
        print(f"Number of unique labels combinations in {study}: {unique_combinations}")
        return unique_combinations

    def get_label_metrics(self, study='train'):
        """Returns label cardinality and label density of a multilabel dataset.
           Label cardinality: average number of labels per entry
           Label density: fraction of assigned labels over total num of labels,
                          averaged per entry
          Parameters:
          labels_df (DataFrame): dataframe of labels
          freq (int): threshold frequency. Labels with a lower value will be filtered.

          Returns:
          DataFrame: filtered labels dataframe

        """
        if study == 'train':
            labels_df = self.train_labels
        elif study == 'val':
            labels_df = self.val_labels
        elif study == 'test':
            labels_df = self.test_labels
        elif study == 'all':
            labels_df = pd.concat([self.train_labels, self.val_labels, self.test_labels])

        sum_labels = labels_df.sum(axis=1)
        total_labels = labels_df.shape[0]
        label_cardinality = 0
        for label in sum_labels:
            label_cardinality += label / total_labels
        label_density = label_cardinality / total_labels
        print("Label cardinality in {}: {}".format(study, label_cardinality))
        print("Label density in {}: {}".format(study, label_density))
        return label_cardinality, label_density


class KunischPlotter:
    def __init__(self, num_lines=30):
        # Plotting linemarks

        linemarks = []
        MARKERS = ['.', '+', 'v', 'x', '*']
        LINE_STYLES = ['-', '--', '-.', ':']

        for i in range(0, num_lines):
            linestyle = LINE_STYLES[random.randint(0, len(LINE_STYLES) - 1)]
            marker = MARKERS[random.randint(0, len(MARKERS) - 1)]
            linemarks.append(linestyle + marker)

        self.linemarks = linemarks

    def plot_results(self, x, score=[], label=[], title="", xlabel="", ylabel="", width=7, height=9, ylim=0.6):
        assert len(x) == len(score[0])
        fig = plt.figure(1)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ylim)
        for i in range(0, len(score)):
            ax.plot(x, score[i], self.linemarks[i], label=label[i])
        ax.legend()
        fig.show()

    def print_confusion_matrix(self, cm, axes, class_label, class_names, fontsize=14, normalize=True):
        df_cm = pd.DataFrame(
            cm, index=class_names, columns=class_names,
        )
        if normalize:
            # print(df_cm)
            df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
            # print(df_cm)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cbar=False, ax=axes, cmap='Blues',
                              annot_kws={"size": 12})
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title(class_label)

    def plot_multiple_matrix(self, cfs_matrix, present_labels, nrows=5, ncols=5, figsize=(6, 10), filename="cm",
                             normalize=True):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

        for axes, cfs_vals, label in zip(ax.flatten(), cfs_matrix, present_labels):
            self.print_confusion_matrix(cfs_vals, axes, label, ["N", "Y"], normalize=normalize)

        fig.tight_layout()
        plt.show()
        # plt.savefig(fig, filename + ".png")
