#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # Export DATA
# ## Seminario de Tesis II, Primavera 2022 
# ### Master of Data Science, Universidad de Chile.
# #### Supervisor: Prof. Benjamín Bustos, Prof. Iván Sipirán
# #### Author: Matías Vergara
# 
# ### References:
# - [SLEEC Homepage](http://manikvarma.org/code/SLEEC/download.html)
# - [SLEEC Paper: Sparse Local Embeddings for Extreme Multi-label Classification](https://papers.nips.cc/paper/2015/hash/35051070e572e47d2c26c241ab88307f-Abstract.html)
# - [The Emerging Trends of Multi-Label Learning](https://arxiv.org/abs/2011.11197)
# - [GitHub: C2AE Multilabel Classification](https://github.com/dhruvramani/C2AE-Multilabel-Classification)
# - 'Learning Deep Latent Spaces for Multi-Label Classfications' published in AAAI 2017
# 
# Este notebook tiene por finalidad exportar un conjunto de datos (a seleccionar con las celdas de selección habituales) al formato requerido por implementaciones oficiales de los modelos presentados en el paper de Emerging Trends of Multi-Label Learning. Hasta el momento se ha experimentado con:
# - SLEEC, sin mayor éxito (requiere una instalación particular de MeX que viene con Matlab 2017b, al cual no logré acceder)
# - C2AE, con resultados mediocres.
# 

# ## Mounting Google Drive

# In[1]:


root_dir = '..'


# ## Imports

# In[2]:


from IPython.display import display
import os
import math
import random
import shutil
import pickle

# Data treatment
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.io import savemat
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from utils import KunischPruner
from utils import KunischMetrics


# ## Dataset and model selection

# In[3]:


LABELS_IN_STUDY = 26 # top N labels will be exported to Matlab

# In[5]:


USE_RN50 = False
SUBCHAPTERS = False
DSFLAGS = [['base'],
           ['blur'],
           ['blur', 'rain', 'ref', 'rot', 'crop', 'elastic'],
           ['blur', 'rain', 'ref', 'rot', 'crop', 'elastic', 'randaug'],
           ['blur', 'rain', 'ref', 'rot', 'elastic'],
           ['crop'],
           ['elastic'],
           ['gausblur'],
           ['mtnblur'],
           ['rain'],
           ['rain_ref_rot'],
           ['rain', 'ref', 'rot', 'elastic'],
           ['randaug'],
           ['ref'],
           ['ref', 'rot'],
           ['rot']]
Ks = [0, 1, 2, 3]
CROP_TIMES = 1
RANDOM_TIMES = 1
ELASTIC_TIMES = 1
GAUSBLUR_TIMES = 1


for DS_FLAGS in DSFLAGS:
    for K in Ks:
        # This cells builds the data_flags variable, that will be used
        # to map the requested data treatment to folders
        MAP_TIMES = {'crop': CROP_TIMES,
                 'randaug': RANDOM_TIMES,
                 'elastic': ELASTIC_TIMES,
                 'gausblur': GAUSBLUR_TIMES
        }

        DS_FLAGS = sorted(DS_FLAGS)
        data_flags = '_'.join(DS_FLAGS) if len(DS_FLAGS) > 0 else 'base'
        MULTIPLE_TRANSF = ['crop', 'randaug', 'elastic', 'gausblur']
        COPY_FLAGS = DS_FLAGS.copy()

        for t in MULTIPLE_TRANSF:
            if t in DS_FLAGS:
                COPY_FLAGS.remove(t)
                COPY_FLAGS.append(t + str(MAP_TIMES[t]))
                data_flags = '_'.join(COPY_FLAGS)

        subchapter_str = 'subchapters' if SUBCHAPTERS else ''
        patterns_dir = os.path.join(root_dir, 'patterns', subchapter_str + data_flags, str(K))
        labels_dir = os.path.join(root_dir, 'labels', subchapter_str + data_flags, str(K))

        print(labels_dir)
        if not (os.path.isdir(patterns_dir) and os.path.isdir(labels_dir)):
            raise FileNotFoundError("No existen directorios de datos para el conjunto de flags seleccionado. Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation {}".format(
                (os.path.isdir(patterns_dir), os.path.isdir(labels_dir))))
        print("Patterns set encontrado en {}".format(patterns_dir))
        print("Labels set encontrado en {}".format(labels_dir))


        # In[7]:


        train_filename = "augmented_train_df.json"
        val_filename = "val_df.json"
        test_filename = "test_df.json"


        # In[8]:

        labels_train = pd.read_json(os.path.join(labels_dir, train_filename), orient='index')
        labels_val = pd.read_json(os.path.join(labels_dir, val_filename), orient='index')
        labels_test = pd.read_json(os.path.join(labels_dir, test_filename), orient='index')


        # In[9]:


        pruner = KunischPruner(LABELS_IN_STUDY)
        with open(os.path.join('../..', 'labels', f'top_{LABELS_IN_STUDY}L.pickle'), 'rb') as f:
            top_labels = pickle.load(f)
        pruner.set_top_labels(top_labels)

        features_dir = os.path.join(root_dir, 'features',
                                    'patterns', data_flags, f'K{str(K)}')
        os.makedirs(features_dir, exist_ok=True)


        # In[16]:


        Y_train = pruner.filter_df(labels_train) # reduce labels to most freq
        Y_test = pruner.filter_df(labels_test) # in both train and test
        Y_val = pruner.filter_df(labels_val)

        images_train = {}
        images_val = {}
        images_test = {}
        datasets = {'train': images_train,
                    'val': images_val,
                    'test': images_test}

        # cargar imagenes con indice como llave
        # hacer dataframe
        # ordenar indices en labels y imagenes para que queden en la misma relacion
        # desordenarlos de alguna forma consistente
        # guardarlos
        for dataset in  datasets.keys():
            print(dataset)
            for chapter in os.listdir(os.path.join(patterns_dir, dataset)):
                for file in os.listdir(os.path.join(patterns_dir, dataset, chapter)):
                    img = cv2.imread(os.path.join(patterns_dir, dataset, chapter, file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (40, 40))
                    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
                    img = img.flatten()
                    img_name = file.split('.')[0]
                    datasets[dataset][img_name] = img

        print(len(images_train.values()))


        # In[17]:


        df_train = pd.DataFrame.from_dict(images_train, orient='index')
        df_val = pd.DataFrame.from_dict(images_val, orient='index')
        df_test = pd.DataFrame.from_dict(images_test, orient='index')


        # In[18]:


        labels_train = Y_train.sort_index()
        df_train = df_train.sort_index()

        labels_val = Y_val.sort_index()
        df_val = df_val.sort_index()

        labels_test = Y_test.sort_index()
        df_test = df_test.sort_index()

        idx = np.random.permutation(labels_train.index)
        labels_train = labels_train.reindex(idx)
        df_train = df_train.reindex(idx)

        idx = np.random.permutation(labels_val.index)
        labels_val = labels_val.reindex(idx)
        df_val = df_val.reindex(idx)

        idx = np.random.permutation(labels_test.index)
        labels_test = labels_test.reindex(idx)
        df_test = df_test.reindex(idx)

        df_train.to_json(os.path.join(features_dir, train_filename), orient='index')
        df_val.to_json(os.path.join(features_dir, val_filename), orient='index')
        df_test.to_json(os.path.join(features_dir, test_filename), orient='index')

