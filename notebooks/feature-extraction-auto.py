#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # Kunisch Features from ResNet architectures
# ## Seminario de Tesis II, Primavera 2022
# ### Master of Data Science. Universidad de Chile.
# #### Prof. guía: Benjamín Bustos - Prof. coguía: Iván Sipirán
# #### Autor: Matías Vergara
# 

# ## Imports

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import os
import copy
import pandas as pd
import math
import random
import shutil

from torch.utils.data import Dataset
from PIL import Image

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np, scipy.io
import argparse
import json


# ## Mounting Google Drive

# In[9]:


try:
    from google.colab import drive
    drive.mount('/content/drive')
    root_dir = 'drive/MyDrive/TesisMV/'
except:
    root_dir = '../'


# ## Dataset and model selection

# In[18]:


#modify only this cell
USE_RN50 = True
SUBCHAPTERS = False

FLAGS = [
    ['base'],
    #['ref'],
    #['rot'],
    #['rain'],
    #['elastic'],
    #['blur'],
    #['gausblur'],
    #['mtnblur'],
    #['crop'],
    #['randaug'],
    ['ref', 'rot'],
    ['ref', 'rot', 'rain'],
    #['ref', 'rot', 'rain', 'elastic'],
    #['ref', 'rot', 'rain', 'elastic', 'blur'],
    #['ref', 'rot', 'rain', 'elastic', 'blur', 'crop'],
    #['ref', 'rot', 'rain', 'elastic', 'blur', 'crop', 'randaug']
]

for DS_FLAGS in FLAGS:

    CROP_TIMES = 1
    RANDOM_TIMES = 1
    ELASTIC_TIMES = 1
    GAUSBLUR_TIMES = 1
    K = 4


    # In[23]:


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

    subchapter_str = 'subchapters/' if SUBCHAPTERS else ''
    Kfolds = {}

    for i in range(0, K):
        print("Fold ", i)
        patterns_dir = os.path.join(root_dir, 'patterns', subchapter_str + data_flags, str(i))
        labels_dir = os.path.join(root_dir, 'labels', subchapter_str + data_flags, str(i))
        # models_path = folder_path + 'models/' + subchapter_str + (f'resnet50_{data_flags}.pth' if USE_RN50 else f'resnet18_{data_flags}.pth')
        # features_path = folder_path + 'features/' + subchapter_str + (f'resnet50_{data_flags}/' if USE_RN50 else f'resnet18_{data_flags}/')
        #rn = 50
        #ep = 65
        #models_path = folder_path + f"models/resnet{rn}_blur_each5/resnet{rn}_blur_e{ep}.pth"
        #features_path = folder_path + f"features/resnet{rn}_blur_each5/resnet{rn}_blur_e{ep}/"
        rn = 50 if USE_RN50 else 18
        models_path = os.path.join(root_dir, 'models', 'resnet', data_flags, f'resnet{rn}_K0.pth')
        features_dir = os.path.join(root_dir, 'features', 'resnet', data_flags, f'resnet{rn}_K{i}')

        if not (os.path.isdir(patterns_dir) and os.path.isdir(labels_dir)):
            print(patterns_dir)
            print(labels_dir)
            raise FileNotFoundError("""
            No existen directorios de datos para el conjunto de flags seleccionado. 
            Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation.
            """)
        if not (os.path.isfile(models_path)):
            print(models_path)
            raise FileNotFoundError(f"""
            No se encontró modelo para el conjunto de flags seleccionado. 
            Verifique que el modelo exista y, de lo contrario, llame a ResNet Retraining
            """)

        Kfolds[i] = {
            'patterns_dir': patterns_dir,
            'labels_dir': labels_dir,
            'model_path': models_path,
            'features_dir': features_dir
        }

        print("--Pattern set encontrado en {}".format(patterns_dir))
        print("--Labels set encontrado en {}".format(labels_dir))
        print("--Modelo encontrado en {}".format(models_path))
        print("--Features a guardar en {}".format(features_dir))


    # ## Dataset loader

    # In[21]:


    class PatternDataset(Dataset):
        def __init__(self, root_dir, transform=None, build_classification=False, name_cla='output.cla'):
            self.root_dir = root_dir
            self.transform = transform
            self.namefiles = []


            self.classes = sorted(os.listdir(self.root_dir))

            for cl in self.classes:
                for pat in os.listdir(os.path.join(self.root_dir, cl)):
                    self.namefiles.append((pat, cl))

            print(f'Files:{len(self.namefiles)}')
            self.namefiles = sorted(self.namefiles, key = lambda x: x[0])

            if build_classification:
                dictClasses = dict()

                for cl in self.classes:
                    dictClasses[cl] = []

                for index, (name, cl) in enumerate(self.namefiles):
                    dictClasses[cl].append((name, index))

                with open(name_cla, 'w') as f:
                    f.write('PSB 1\n')
                    f.write(f'{len(self.classes)} {len(self.namefiles)}\n')
                    f.write('\n')
                    for cl in self.classes:
                        f.write(f'{cl} 0 {len(dictClasses[cl])}\n')
                        for item in dictClasses[cl]:
                            f.write(f'{item[1]}\n')
                        f.write('\n')

        def __len__(self):
            return len(self.namefiles)

        def __getitem__(self, index):
            if torch.is_tensor(index):
                index = index.tolist()

            img_name = os.path.join(self.root_dir, self.namefiles[index][1], self.namefiles[index][0])
            image = Image.open(img_name).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return self.namefiles[index], image


    # ## Funciones auxiliares

    # In[22]:


    def imshow(inp, title = None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        plt.imshow(inp)
        plt.show()

    def get_vector(model,layer, dim_embedding, x):

      my_embedding = torch.zeros(dim_embedding)

      def copy_data(m,i,o):
        my_embedding.copy_(o.data.squeeze())

      h = layer.register_forward_hook(copy_data)
      model(x)
      h.remove()

      return my_embedding


    # ## Extraction

    # In[24]:


    DEVICE = 0
    # 0 3090 (o 1060 en local)
    # 1 y 2 2080


    # In[25]:


    random.seed(30)

    for i in range(0, K):
        patterns_dir = Kfolds[i]['patterns_dir']
        labels_dir = Kfolds[i]['labels_dir']
        model_path = Kfolds[i]['model_path']
        features_dir = Kfolds[i]['features_dir']

        output_train = os.path.join(features_dir, "augmented_train_df.json")
        output_val = os.path.join(features_dir, "val_df.json")
        output_test = os.path.join(features_dir, "test_df.json")

        train_df = pd.read_json(os.path.join(labels_dir, "augmented_train_df.json"), orient='index')
        val_df = pd.read_json(os.path.join(labels_dir, "val_df.json"), orient='index')
        test_df = pd.read_json(os.path.join(labels_dir, "test_df.json"), orient='index')

        train_pts = train_df.index.values
        val_pts = val_df.index.values
        test_pts = test_df.index.values

        device = ('cuda:0' if torch.cuda.is_available() else None)
        if device is None:
            raise Exception("La GPU solicitada no está disponible")

        my_transform = transforms.Compose([ transforms.Resize(224),
                                            #transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
                            ])

        dataTrain = PatternDataset(root_dir=os.path.join(patterns_dir, 'train'), transform=my_transform)
        dataVal = PatternDataset(root_dir=os.path.join(patterns_dir, 'val'), transform=my_transform)
        dataTest = PatternDataset(root_dir=os.path.join(patterns_dir, 'test'), transform=my_transform)

        loaderTrain = DataLoader(dataTrain)
        loaderVal = DataLoader(dataVal)
        loaderTest = DataLoader(dataTest)

        if USE_RN50:
            model = models.resnet50(pretrained = True)
        else:
            model = models.resnet18(pretrained = True)
        dim = model.fc.in_features

        output_dim = 20 if SUBCHAPTERS else 6
        model.fc = nn.Linear(dim, output_dim)

        model = model.to(device)

        try:
            model.load_state_dict(torch.load(models_path))
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

        layer = model._modules.get('avgpool')

        model.eval()

        #features = []
        features_train = {}
        features_val = {}
        features_test = {}


        for name, img in loaderTrain:
          feat = get_vector(model, layer, dim, img.to(device))
          namefile = name[0][0]
          code, rest = namefile.split('.')
          features_train[code] = feat.numpy().tolist()
          #features.append(feat.numpy())

        for name, img in loaderVal:
          feat = get_vector(model, layer, dim, img.to(device))
          namefile = name[0][0]
          code, rest = namefile.split('.')
          features_val[code] = feat.numpy().tolist()
          #features.append(feat.numpy())

        for name, img in loaderTest:
          feat = get_vector(model, layer, dim, img.to(device))
          namefile = name[0][0]
          code, rest = namefile.split('.')
          features_test[code] = feat.numpy().tolist()
        #features = np.vstack(features)
        #print(features.shape)

        os.makedirs(features_dir, exist_ok=True)

        features_train_df = pd.DataFrame.from_dict(features_train, orient='index')
        features_val_df = pd.DataFrame.from_dict(features_val, orient='index')
        features_test_df = pd.DataFrame.from_dict(features_test, orient='index')

        features_train_df.to_json(output_train, orient='index')
        features_val_df.to_json(output_val, orient='index')
        features_test_df.to_json(output_test, orient='index')


