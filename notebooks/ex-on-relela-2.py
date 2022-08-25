#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # Resnet Retraining
# ## Seminario de Tesis II, Primavera 2022
# ### Master of Data Science. Universidad de Chile.
# #### Prof. guía: Benjamín Bustos - Prof. coguía: Iván Sipirán
# #### Autor: Matías Vergara

# ## Imports

# In[13]:


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
import warnings


# ## Mounting Google Drive

# In[14]:


try:
    from google.colab import drive
    drive.mount('/content/drive')
    folder_path = 'drive/MyDrive/TesisMV/'
except:
    folder_path = '../'


# ## Dataset and model selection

# In[15]:


#modify only this cell
USE_RN50 = False
SUBCHAPTERS = False

FLAGS = [
    ['ref'],
    ['rot'],
    ['rain'],
    ['elastic'],
    ['blur'],
    ['gausblur'],
    ['mtnblur'],
    ['crop'],
    ['randaug'],
    ['ref', 'rot'],
    ['ref', 'rot', 'rain'],
    ['ref', 'rot', 'rain', 'elastic'],
    ['ref', 'rot', 'rain', 'elastic', 'blur'],
    ['ref', 'rot', 'rain', 'elastic', 'blur', 'crop'],
    ['ref', 'rot', 'rain', 'elastic', 'blur', 'crop', 'randaug']
]

for DS_FLAGS in FLAGS:
    warnings.warn(f"Iniciando DS_FLAGS {DS_FLAGS}")
              # 'ref': [invertX, invertY],
              # 'rot': [rotate90, rotate180, rotate270],
              # 'crop': [crop] * CROP_TIMES,
              # 'blur': [blur],
              # 'emboss': [emboss],
              # 'randaug': [randaug],
              # 'rain': [rain],
              # 'elastic': [elastic]
    CROP_TIMES = 1
    RANDOM_TIMES = 1
    ELASTIC_TIMES = 1
    GAUSBLUR_TIMES = 1
    SAVE_EACH = -1 # -1 to save only the best model
    TRAINING_EPOCHS = 80
    K = 4
    k_model = 0


    # In[17]:


    # This cells builds the data_flags variable, that will be used
    # to map the requestes data treatment to folders
    MAP_TIMES = {'crop': CROP_TIMES,
             'randaug': RANDOM_TIMES,
             'elastic': ELASTIC_TIMES,
             'gausblur': GAUSBLUR_TIMES,
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
    patterns_path = folder_path + 'patterns/' + subchapter_str + data_flags + '/' + str(k_model)
    labels_path = folder_path + 'labels/' + subchapter_str + data_flags + '/' + str(k_model)
    if not (os.path.isdir(patterns_path) and os.path.isdir(labels_path)):
        raise FileNotFoundError("No existen directorios de datos para el conjunto de flags seleccionado. Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation")
    print("Pattern set encontrado en {}".format(patterns_path))
    print("Labels set encontrado en {}".format(labels_path))
    OUTPUT_FILENAME = f'resnet50_K{k_model}.pth' if USE_RN50 else f'resnet18_K{k_model}.pth'


    # In[19]:


    model_output_dir = folder_path + 'models/resnet/{}/{}'.format(data_flags, 'subchapters/' if SUBCHAPTERS else '')
    model_output_path = model_output_dir + OUTPUT_FILENAME
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"El modelo resultante se guardará en {model_output_dir}")


    # ## Transfer Learning

    # In[22]:


    pathDataset = patterns_path + '/'

    train_dataset = torchvision.datasets.ImageFolder(pathDataset + 'train',
                                                        transform = transforms.Compose([
                                                            transforms.RandomVerticalFlip(),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomResizedCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std = [0.229, 0.224, 0.225])]))

    val_dataset = torchvision.datasets.ImageFolder(pathDataset + 'val',
                                                        transform = transforms.Compose([ transforms.Resize(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std = [0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    class_names = train_dataset.classes

    device = ('cuda:0' if torch.cuda.is_available() else None)
    if device is None:
        raise Exception("La gpu solicitada no esta disponible")
    print("Usando ", device)

    def train_model(model, criterion, optimizer, num_epochs=30, output_path = 'model.pth', save_each = -1, patience=15):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        bad_epochs = 0
        previous_loss = 9999
        best_epoch = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs-1}')
            print('-' * 10)

            model.train()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds ==  labels.data)

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset)

            print('Train Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            #Validation
            model.eval()
            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_dataset)
            epoch_acc = running_corrects / len(val_dataset)
            print('Val Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if save_each > -1 and epoch%save_each == 0:
                path = output_path.split("/")
                filename =  path[-1]
                epoch_filename =filename.split(".")[0] + "_e" + str(epoch) + "." + filename.split(".")[1]
                new_path = path[:-1]
                new_path.append(epoch_filename)
                new_path = '/'.join(new_path)
                torch.save(model.state_dict(), new_path)
                print("Saving model at epoch {} as {}".format(epoch, new_path))

            if epoch_loss < previous_loss:
                previous_loss = epoch_loss
                bad_epochs = 0
                best_epoch = epoch
            else:
                bad_epochs += 1
                if bad_epochs == patience:
                    print(f"Se agotó la paciencia. Mejor época: {best_epoch}.")
                    break

        print('Best accuracy: {:.4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)

        return model


    # In[23]:


    device


    # In[24]:


    if USE_RN50:
        model_ft = models.resnet50(pretrained=True)
    else:
        model_ft = models.resnet18(pretrained=True)
    num_ft = model_ft.fc.in_features

    output_dim = 20 if SUBCHAPTERS else 6
    model_ft.fc = nn.Linear(num_ft, output_dim)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    groups = [{'params': model_ft.conv1.parameters(),'lr':learning_rate/4},
                {'params': model_ft.bn1.parameters(),'lr':learning_rate/4},
                {'params': model_ft.layer1.parameters(),'lr':learning_rate/4},
                {'params': model_ft.layer2.parameters(),'lr':learning_rate/2},
                {'params': model_ft.layer3.parameters(), 'lr':learning_rate/2},
                {'params': model_ft.layer4.parameters(),'lr':learning_rate},
                {'params': model_ft.fc.parameters(), 'lr':learning_rate}]

    optimizer = torch.optim.Adam(model_ft.parameters(), lr = 0.0015)

    output_path = model_output_path

    # change save_each and output_path to get partial outputs
    model_ft = train_model(model_ft, criterion, optimizer, num_epochs=TRAINING_EPOCHS,
                           save_each=SAVE_EACH, output_path=output_path)

    # save best model
    torch.save(model_ft.state_dict(), output_path)


    # ## Testing Transfer Learning

    # In[ ]:


    model = model_output_path
    # model = '../' + 'models/resnet/resnet50_blur_each5/resnet50_blur_e75.pth'
    #USE_RN50 = True

    pathDataset = patterns_path + '/'

    test_dataset = torchvision.datasets.ImageFolder(pathDataset + 'test',
                                                        transform = transforms.Compose([ transforms.Resize(224),
                                                                        #transforms.CenterCrop(224),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                            std = [0.229, 0.224, 0.225])]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    if USE_RN50:
        model_ft = models.resnet50(pretrained=True)
    else:
        model_ft = models.resnet18(pretrained=True)

    output_dim = 20 if SUBCHAPTERS else 6
    model_ft.fc = nn.Linear(num_ft, output_dim)

    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(model))
    criterion = nn.CrossEntropyLoss()

    model_ft.eval()
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects / len(test_dataset)
    print('Test Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    # In[ ]:


    data_flags


    # In[ ]:




