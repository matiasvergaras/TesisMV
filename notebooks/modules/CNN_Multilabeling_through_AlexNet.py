#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # CNN Multilabeling through AlexNet
# ## Seminario de Tesis II, Primavera 2022
# ### Master of Data Science. Universidad de Chile.
# #### Prof. guía: Benjamín Bustos - Prof. coguía: Iván Sipirán
# #### Autor: Matías Vergara
# 
# El objetivo de este notebook es realizar predicciones multilabel sobre patrones geométricos mediante AlexNet.

# ## Montando Google Drive

# In[1]:


# Mounting google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    root_dir = '../content/gdrive/MyDrive'
except:
    root_dir = '../../'


# In[2]:


import os
import pickle
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import math
from torch.nn.utils.rnn import pack_padded_sequence
from matplotlib import pyplot as plt


# ## Selección de dataset y experimento

# In[ ]:

experimentos = [
    {
        'DS_FLAGS': ['ref'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['rot'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['blur'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['crop'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['elastic'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['rain'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['randaug'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['crop'],
        'CROP_TIMES': 2,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['elastic'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 2,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['randaug'],
        'CROP_TIMES': 2,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 2,
        'RANDOM_TIMES': 2
    },
    {
        'DS_FLAGS': ['ref', 'rot', 'blur'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['ref', 'rot', 'crop'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['ref', 'rot', 'blur', 'crop'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['crop', 'elastic', 'randaug'],
        'CROP_TIMES': 2,
        'ELASTIC_TIMES': 2,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 2
    },
    {
        'DS_FLAGS': ['crop', 'elastic', 'blur'],
        'CROP_TIMES': 3,
        'ELASTIC_TIMES': 3,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['ref', 'rot', 'blur', 'rain', 'crop', 'elastic'],
        'CROP_TIMES': 3,
        'ELASTIC_TIMES': 3,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['randaug'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 1,
        'RANDOM_TIMES': 10
    },
    {
        'DS_FLAGS': ['gausblur'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 4,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['gausblur'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 2,
        'RANDOM_TIMES': 1
    },
    {
        'DS_FLAGS': ['gaus', 'mtnblur', 'elastic'],
        'CROP_TIMES': 1,
        'ELASTIC_TIMES': 1,
        'GAUSBLUR_TIMES': 2,
        'RANDOM_TIMES': 1
    },
]

evoluciones = [
    {
        'DS_FLAGS': ['blur'],
        'use_pos_weights': True,
    }
]

IS_EVO = True
for evolucion in evoluciones:
    for i in range(26, 300, 10):
        DS_FLAGS = evolucion['DS_FLAGS']
        SUBCHAPTERS = False
        CROP_TIMES = 1
        RANDOM_TIMES = 1
        ELASTIC_TIMES = 1
        GAUSBLUR_TIMES = 1

        use_pos_weights = evolucion['use_pos_weights']
        pos_weights_factor = 1
        NUM_LABELS = i
        use_testval = True
        BATCH_SIZE = 124

        TH_TRAIN = 0.5
        TH_VAL = 0.5
        TH_TEST = 0.5

        # 0 es 3090, 1 y 2 son 2080
        CUDA_ID = 0


        # In[ ]:


        # This cells builds the data_flags variable, that will be used
        # to map the requested data treatment to folders
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

        patterns_path = os.path.join(root_dir, 'patterns', data_flags)
        labels_path = os.path.join(root_dir, 'labels', data_flags)

        if not (os.path.isdir(patterns_path) and os.path.isdir(labels_path)):
            raise FileNotFoundError("No existen directorios de datos para el conjunto de flags seleccionado. Verifique que el dataset exista en {}".format(
                (os.path.isdir(patterns_path), os.path.isdir(labels_path))))
        print("Patterns set encontrado en {}".format(patterns_path))
        print("Labels set encontrado en {}".format(labels_path))

        exp_name = f"{NUM_LABELS}L"
        exp_name += "_testval" if use_testval else ""
        weights_str = str(pos_weights_factor)
        weights_str = weights_str.replace('.','_')
        exp_name += f'_weighted_{weights_str}' if use_pos_weights else ''
        print(f"Nombre del experimento: {exp_name}")

        output_dir = os.path.join(root_dir, "outputs", "alexnet", data_flags, exp_name)
        if IS_EVO:
            output_dir = os.path.join(root_dir, "outputs", "alexnet", data_flags, 'evo', exp_name)
        os.makedirs(output_dir, exist_ok = True)

        model_dir = os.path.join(root_dir, "models", "alexnet", data_flags)
        os.makedirs(model_dir, exist_ok = True)
        model_path = os.path.join(model_dir, exp_name + '.pth')

        print(f"Los resultados se guardarán en: {output_dir}")
        print(f"Los resultados se guardarán en: {output_dir}")


        # In[ ]:


        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        available_gpus


        # In[ ]:


        labels_train = pd.read_json(os.path.join(labels_path, 'augmented_train_df.json'), orient='index')
        labels_val = pd.read_json(os.path.join(labels_path, 'val_df.json'), orient='index')
        labels_test = pd.read_json(os.path.join(labels_path, 'test_df.json'), orient='index')


        # In[ ]:


        labels_test


        # In[ ]:


        def filter_labels(labels_df, freq=25, number_labels = None):
          """Filters a label dataframe based on labels frequency (number of events)

            Parameters:
            labels_df (DataFrame): dataframe of labels
            freq (int): threshold frequency. Labels with a lower value will be filtered.

            Returns:
            DataFrame: filtered labels dataframe

          """
          top_labels = None

          if not number_labels:
            filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > freq]
            top_labels = filtered_df.sum().sort_values(ascending=False)
            return top_labels, 0

          if number_labels:
              filtered_labels = labels_df.shape[1]
              pivot = 0
              while filtered_labels > number_labels:
                #print(filtered_labels, number_labels, pivot)
                filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pivot]
                top_labels = filtered_df.sum().sort_values(ascending=False)
                filtered_labels = filtered_df.shape[1]
                pivot += 1
              print("Aplicando threshold {} para trabajar con {} labels".format(pivot, len(top_labels.values)))
              return top_labels, pivot

        def filter_dfs(df, top_labels_df):
          df = df[df.columns.intersection(top_labels_df.index)]
          return df

        def make_positive_weights(labels, factor=1):
            total = labels.values.sum()
            weights = [0.] * len(labels)
            for i, label in enumerate(labels):
              weights[i] = total/(factor * labels[i])
            return weights

        def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
            '''
            Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
            http://stackoverflow.com/q/32239577/395857
            '''
            acc_list = []
            for i in range(y_true.shape[0]):
                set_true = set( np.where(y_true[i])[0] )
                set_pred = set( np.where(y_pred[i])[0] )
                #print('\nset_true: {0}'.format(set_true))
                #print('set_pred: {0}'.format(set_pred))
                tmp_a = None
                if len(set_true) == 0 and len(set_pred) == 0:
                    tmp_a = 1
                else:
                    tmp_a = len(set_true.intersection(set_pred))/                    float( len(set_true.union(set_pred)) )
                #print('tmp_a: {0}'.format(tmp_a))
                acc_list.append(tmp_a)
            return np.mean(acc_list)


        # In[ ]:


        train_labels = pd.read_json(os.path.join(labels_path, 'augmented_train_df.json'), orient='index')
        if not os.path.isfile(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle')):
            print(f"Creando top_labels para {NUM_LABELS} labels")
            top_labels, _ = filter_labels(train_labels, number_labels = NUM_LABELS)
            save = "y" #input(f"Se creará un archivo nuevo para {len(top_labels)} labels. Desea continuar? (y/n)")
            if save == "y":
                with open(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle'), 'wb') as f:
                    pickle.dump(top_labels, f)
                print(top_labels)
            else:
                raise Exception("No se logró cargar top_labels")
        else:
            print(f"Usando top_labels previamente generados para {NUM_LABELS} labels")
            with open(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle'), 'rb') as f:
                top_labels = pickle.load(f)
            #print(top_labels)


        # In[ ]:


        # Device configuration
        device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')
        print(f"Usando device: {torch.cuda.get_device_name(device)}")


        # In[ ]:


        train_labels = pd.read_json(os.path.join(labels_path, 'augmented_train_df.json'), orient='index')
        NUM_LABELS = len(top_labels) # la cantidad final de etiquetas a trabajar

        if use_pos_weights:
            pos_weights = make_positive_weights(top_labels, pos_weights_factor)
            pos_weights = torch.Tensor(pos_weights).float().to(device)
        else:
            pos_weights = None

        # images_dir=os.path.join(root_dir, 'patterns', data_flags, 'train'),
        # labels_file=os.path.join(root_dir, 'labels', data_flags, 'augmented_train_df.json'),
        class KunischDataset(torch.utils.data.Dataset):

          def __init__(self, images_dir, labels_file, transform, top_labels, extra_labels = None, extra_images_dir = None):
            """
            Args:
                text_file(string): path to text file
                root_dir(string): directory with all train images
            """
            self.labels_frame = filter_dfs(pd.read_json(labels_file, orient='index'), top_labels)
            self.num_labels = len(top_labels)
            self.images_dir = images_dir
            self.labels_file = labels_file
            self.transform = transform
            self.flags = data_flags
            self.top_labels = top_labels
            self.extra_images_dir = None

            # para crear conjunto test-val
            if extra_labels:
              extra_labels_frame = filter_dfs(pd.read_json(extra_labels, orient='index'), top_labels)
              self.labels_frame = pd.DataFrame.append(self.labels_frame, extra_labels_frame)
              self.extra_images_dir = extra_images_dir

          def __len__(self):
            return len(self.labels_frame)

          def __getitem__(self, idx):
            img_id = self.labels_frame.iloc[idx].name + '.png'
            img_name = None
            for chapter in os.listdir(self.images_dir):
              if img_id in os.listdir(os.path.join(self.images_dir, chapter)):
                img_name = os.path.join(self.images_dir, chapter, img_id)
                break
            # caso test-val
            if (self.extra_images_dir is not None) and (img_name is None):
              for chapter in os.listdir(self.extra_images_dir):
                if img_id in os.listdir(os.path.join(self.extra_images_dir, chapter)):
                  img_name = os.path.join(self.extra_images_dir, chapter, img_id)
                  break
            if img_name is None:
              raise Exception(f'No se encontró la imagen para {img_id}')
            image = Image.open(img_name)
            image = image.convert('RGB')
            image = self.transform(image)
            labels = self.labels_frame.iloc[idx].values
            labels = np.array(labels)
            labels = torch.from_numpy(labels.astype('int'))
            #print(img_id, img_name, self.labels_frame.iloc[idx], self.labels_frame.iloc[idx].values, labels)
            sample = {'image': image, 'labels': labels, 'paths': img_name}
            return sample


        # Alexnet requires 227 x 227
        # Training
        kunischTrainSet = KunischDataset(images_dir=os.path.join(patterns_path, 'train'),
                                         labels_file=os.path.join(labels_path, 'augmented_train_df.json'),
                                         transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(
                                                                           mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])]),
                                         top_labels=top_labels)

        kunischTrainLoader = torch.utils.data.DataLoader(kunischTrainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Validation
        kunischValidationSet = KunischDataset(images_dir=os.path.join(patterns_path, 'val'),
                                              labels_file=os.path.join(labels_path, 'val_df.json'),
                                              transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(
                                                                                mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]),
                                              top_labels=top_labels)

        kunischValidationLoader = torch.utils.data.DataLoader(kunischValidationSet, batch_size=BATCH_SIZE, shuffle=True,
                                                              num_workers=0)

        # Test
        kunischTestSet = KunischDataset(images_dir=os.path.join(patterns_path, 'test'),
                                        labels_file=os.path.join(labels_path, 'test_df.json'),
                                        transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(
                                                                          mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]),
                                        top_labels=top_labels,
                                        extra_labels=os.path.join(labels_path, 'val_df.json' if use_testval else None),
                                        extra_images_dir=os.path.join(patterns_path, 'val' if use_testval else None))

        kunischTestLoader = torch.utils.data.DataLoader(kunischTestSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


        # Define the function for training, validation, and test
        def alex_train(epoch, num_epochs, train_losses, learning_rate, w):
          alex_net.train()
          train_loss = 0
          TN = 0
          TP = 0
          FP = 0
          FN = 0
          preds_total = np.empty((1, NUM_LABELS), dtype=int)
          labels_total = np.empty((1, NUM_LABELS), dtype=int)

          for i, sample_batched in enumerate(kunischTrainLoader, 1):
              inputs = sample_batched['image'].to(device)
              labels = sample_batched['labels'].to(device)

              # zero the parameter gradients
              optimizer.zero_grad()

              # forward + backward + optimize
              outputs = alex_net(inputs)
              loss = criterion(outputs.float(), labels.float())
              loss.backward()
              optimizer.step()

              train_loss += loss.item()
              pred = (torch.sigmoid(outputs).data > TH_TRAIN).int()
              # print(pred)
              labels = labels.int()
              # print(labels)
              preds_total = np.concatenate((preds_total, pred.cpu()), axis=0)
              labels_total = np.concatenate((labels_total, labels.cpu()), axis=0)

              TP += ((pred == 1) & (labels == 1)).float().sum()  # True Positive Count
              TN += ((pred == 0) & (labels == 0)).float().sum()  # True Negative Count
              FP += ((pred == 1) & (labels == 0)).float().sum()  # False Positive Count
              FN += ((pred == 0) & (labels == 1)).float().sum()  # False Negative Count
              #print('TP: {}\t TN: {}\t FP: {}\t FN: {}\n'.format(TP, TN, FP, FN))


          TP = TP.cpu().numpy()
          TN = TN.cpu().numpy()
          FP = FP.cpu().numpy()
          FN = FN.cpu().numpy()

          accuracy = (TP + TN) / (TP + TN + FP + FN)
          precision = TP / (TP + FP)
          recall = TP / (TP + FN)
          f1_score = 2 * (precision * recall) / (precision + recall)
          train_loss = train_loss / len(kunischTrainLoader.dataset) * BATCH_SIZE
          hs = hamming_score(preds_total, labels_total)
          train_losses.append([epoch, learning_rate, w, train_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])

          # print statistics
          print('Train Trial [{}/{}], LR: {:.4g}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
                .format(epoch, num_epochs, optimizer.param_groups[0]['lr'], w, train_loss, accuracy, f1_score, hs))
          return hs

        def alex_valid(epoch, num_epochs, valid_losses, learning_rate, w):
          # Have our model in evaluation mode
          alex_net.eval()
          # Set losses and Correct labels to zero
          valid_loss = 0
          TN = 0
          TP = 0
          FP = 0
          FN = 0
          preds_total = np.empty((1, NUM_LABELS), dtype=int)
          labels_total = np.empty((1, NUM_LABELS), dtype=int)
          with torch.no_grad():
              for i, sample_batched in enumerate(kunischValidationLoader, 1):
                  inputs = sample_batched['image'].to(device)
                  labels = sample_batched['labels'].to(device)
                  outputs = alex_net(inputs)
                  loss = criterion(outputs.float(), labels.float())
                  valid_loss += loss.item()
                  pred = (torch.sigmoid(outputs).data > TH_VAL).int()
                  labels = labels.int()
                  preds_total = np.concatenate((preds_total, pred.cpu()), axis=0)
                  labels_total = np.concatenate((labels_total, labels.cpu()), axis=0)

                  TP += ((pred == 1) & (labels == 1)).float().sum()  # True Positive Count
                  TN += ((pred == 0) & (labels == 0)).float().sum()  # True Negative Count
                  FP += ((pred == 1) & (labels == 0)).float().sum()  # False Positive Count
                  FN += ((pred == 0) & (labels == 1)).float().sum()  # False Negative Count
                  # print('TP: {}\t TN: {}\t FP: {}\t FN: {}\n'.format(TP,TN,FP,FN) )

              TP = TP.cpu().numpy()
              TN = TN.cpu().numpy()
              FP = FP.cpu().numpy()
              FN = FN.cpu().numpy()
              accuracy = (TP + TN) / (TP + TN + FP + FN)
              precision = TP / (TP + FP)
              recall = TP / (TP + FN)
              f1_score = 2 * (precision * recall) / (precision + recall)
              hs = hamming_score(preds_total, labels_total)

              scheduler.step(hs)

              valid_loss = valid_loss / len(kunischValidationLoader.dataset) * BATCH_SIZE  # 1024 is the batch size
              valid_losses.append(
                  [epoch, learning_rate, w, valid_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])
              # print statistics
              print('Valid Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
                    .format(epoch, num_epochs, optimizer.param_groups[0]['lr'], w, valid_loss, accuracy, f1_score, hs))
              return hs

        def alex_test(epoch, num_epochs, pred_array, test_losses, learning_rate, w, show_images=1):
          # Have our model in evaluation mode
          alex_net.eval()
          # Set losses and Correct labels to zero
          test_loss = 0
          TN = 0
          TP = 0
          FP = 0
          FN = 0
          preds_total = np.empty((1, NUM_LABELS), dtype=int)
          labels_total = np.empty((1, NUM_LABELS), dtype=int)

          with torch.no_grad():
              for i,sample_batched in enumerate(kunischTestLoader, 1):
                  inputs = sample_batched['image'].to(device)
                  labels = sample_batched['labels'].to(device)
                  paths = sample_batched['paths']
                  outputs = alex_net(inputs)

                  loss = criterion(outputs.float(), labels.float())
                  test_loss += loss.item()
                  pred = (torch.sigmoid(outputs).data > TH_TEST).int()
                  # print(pred)
                  labels = labels.int()
                  # print(labels)
                  pred_array.append([paths, test_loss, labels, pred])
                  preds_total = np.concatenate((preds_total, pred.cpu()), axis=0)
                  labels_total = np.concatenate((labels_total, labels.cpu()), axis=0)

                  for j in range(0, min(BATCH_SIZE, show_images)): # j itera sobre ejemplos
                      print(f"Mostrando imagen {j} del batch {i}")
                      img = np.transpose(sample_batched['image'][j]) # imagen j
                      plt.imshow(img, interpolation='nearest')
                      plt.show()
                      labels_correctos = ""
                      labels_predichos = ""
                      for k in range(0, len(pred[j])):
                        labels_correctos += (kunischTestSet.labels_frame.columns.values[k]+' ') if labels[j].cpu().detach()[k] else ""
                        labels_predichos += (kunischTestSet.labels_frame.columns.values[k]+' ') if pred[j].cpu().detach()[k] else ""
                      print("Labels correctos:")
                      #print(labels[j].cpu().detach().numpy())
                      print(labels_correctos)
                      print("Labels predichos:")
                      #print(pred[j].cpu().detach().numpy())
                      print(labels_predichos)
                      print("\n")

                  TP += ((pred == 1) & (labels == 1)).float().sum()  # True Positive Count
                  TN += ((pred == 0) & (labels == 0)).float().sum()  # True Negative Count
                  FP += ((pred == 1) & (labels == 0)).float().sum()  # False Positive Count
                  FN += ((pred == 0) & (labels == 1)).float().sum()  # False Negative Count
                  # print('TP: {}\t TN: {}\t FP: {}\t FN: {}\n'.format(TP,TN,FP,FN) )

              TP = TP.cpu().numpy()
              TN = TN.cpu().numpy()
              FP = FP.cpu().numpy()
              FN = FN.cpu().numpy()
              accuracy = (TP + TN) / (TP + TN + FP + FN)
              precision = TP / (TP + FP)
              recall = TP / (TP + FN)
              f1_score = 2 * (precision * recall) / (precision + recall)
              hs = hamming_score(preds_total, labels_total)
              test_loss = test_loss / len(kunischTestLoader.dataset) * 1024  # 1024 is the batch size
              test_losses.append([epoch, learning_rate, w, test_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])
              # print statistics
              print('Test Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
                    .format(epoch, num_epochs, optimizer.param_groups[0]['lr'], w, test_loss, accuracy, f1_score, hs))
              return hs


        # In[ ]:


        def filter_labels(labels_df, freq=25, number_labels = None):
          """Filters a label dataframe based on labels frequency (number of events)

            Parameters:
            labels_df (DataFrame): dataframe of labels
            freq (int): threshold frequency. Labels with a lower value will be filtered.

            Returns:
            DataFrame: filtered labels dataframe

          """
          top_labels = None

          if not number_labels:
            filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > freq]
            top_labels = filtered_df.sum().sort_values(ascending=False)
            return top_labels, 0

          if number_labels:
              filtered_labels = labels_df.shape[1]
              pivot = 0
              while filtered_labels > number_labels:
                #print(filtered_labels, number_labels, pivot)
                filtered_df = labels_df.loc[:, labels_df.sum(axis=0) > pivot]
                top_labels = filtered_df.sum().sort_values(ascending=False)
                filtered_labels = filtered_df.shape[1]
                pivot += 1
              print("Aplicando threshold {} para trabajar con {} labels".format(pivot, len(top_labels.values)))
              return top_labels, pivot

        def filter_dfs(df, top_labels_df):
          df = df[df.columns.intersection(top_labels_df.index)]
          return df

        def make_positive_weights(labels, factor=1):
            total = labels.values.sum()
            weights = [0.] * len(labels)
            for i, label in enumerate(labels):
              weights[i] = total/(factor * labels[i])
            return weights


        # In[ ]:


        # Hyper Parameter Tuning
        alex_net = models.alexnet(pretrained=True)
        for param in alex_net.parameters():
            param.requires_grad = False
        alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)

        train_losses = []
        validation_losses = []
        num_epochs = 10

        for epoch in range(num_epochs):
          learning_rate = round(np.exp(random.uniform(np.log(.0001), np.log(.01))), 4)  # pull geometrically
          w = round(np.exp(random.uniform(np.log(3.1e-7), np.log(3.1e-5))), 10)  # pull geometrically

          # Reset Model per test
          alex_net = models.alexnet(pretrained=True)
          alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)
          alex_net.to(device)

          optimizer = torch.optim.Adam(alex_net.parameters(), lr=learning_rate, weight_decay=w)
          criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
          scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=5, min_lr=0.00005)

          train_losses_df = pd.DataFrame(train_losses)
          train_losses_df.to_csv(os.path.join(output_dir, 'loss_hypertrain.csv'))

          alex_valid(epoch, num_epochs, validation_losses, learning_rate, w)
          validation_losses_df = pd.DataFrame(validation_losses)
          validation_losses_df.to_csv(os.path.join(output_dir, 'loss_hyperval.csv'))


        # In[ ]:


        # Training
        from torch.optim.lr_scheduler import StepLR
        train_losses = []
        validation_losses = []
        num_epochs = 200
        learning_rate = 0.001
        w = 0.01

        # Early Stopping
        patience = 10
        bad_epochs = 0
        best_score = 0.0
        best_weights = None

        alex_net = models.alexnet(pretrained=True)
        alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)
        alex_net.to(device)

        optimizer = torch.optim.Adam(alex_net.parameters(), lr=learning_rate, weight_decay=w)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=5, min_lr=0.0001)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        for epoch in range(num_epochs):
          score_train = alex_train(epoch, num_epochs, train_losses, learning_rate, w)
          score_valid = alex_valid(epoch, num_epochs, validation_losses, learning_rate, w)
          print("")

          # Early Stopping
          if score_valid > best_score:
            bad_epochs = 0
            best_epoch = epoch
            best_score = score_valid
            best_weights = alex_net.state_dict()
          else:
            bad_epochs += 1

          if bad_epochs == patience:
            print("Out of patience!")
            print(f"Best epoch: {best_epoch}")
            break

        torch.save(best_weights, model_path)


        # In[ ]:


        # alerta sonora para cuando haya terminado de entrenar
        import winsound
        import time
        for i in range(0, 5):
            duration = 500
            freq = 1500
            winsound.Beep(freq, duration)
            time.sleep(0.5)
            i+=1
        winsound.Beep(freq, 2000)



        # In[ ]:


        # Testing
        test_losses = []
        test_pred = []
        learning_rate = 0.0001
        w = 0.001

        # Reset Model
        alex_net = models.alexnet(pretrained=True)
        alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)
        alex_net.load_state_dict(torch.load(model_path))
        alex_net.to(device)

        optimizer = torch.optim.Adam(alex_net.parameters(), lr=learning_rate, weight_decay=w)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        alex_test(1, 1, test_pred, test_losses, learning_rate, w, show_images = 0)
        test_pred_df = pd.DataFrame(test_pred)


        # In[52]:


        preds = test_pred[0][3].cpu().detach().numpy()
        for i in range(1, len(test_pred)):
          pbi = test_pred[i][3].cpu().detach().numpy()
          preds = np.concatenate((preds, pbi), axis=0)
        save_df = pd.DataFrame(preds)
        save_df.to_csv(os.path.join(output_dir, 'predictions.csv'))
        print(f"Predicciones guardadas en {os.path.join(output_dir, 'predictions.csv')}")

        preds = pd.read_csv(os.path.join(output_dir, 'predictions.csv'), index_col=0)
        preds = preds.values
        testval = pd.concat([labels_test, labels_val])
        testval = filter_dfs(testval, top_labels)

        print(preds.shape)
        print(testval.shape)
        hs_final = hamming_score(preds, testval.values)
        print(f"HS final, NUM_LABELS: {hs_final}, {NUM_LABELS}")


        # In[53]:


        metadata = {
            'BATCH_SIZE': BATCH_SIZE,
            'optimizer': (type (optimizer).__name__),
            'scheduler': (type (scheduler).__name__),
            'criterion': (type (criterion).__name__),
            'epochs': num_epochs,
            'best_epoch': best_epoch,
            'data_flags': data_flags,
            'use_pos_weights': use_pos_weights,
            'pos_weights_factor': pos_weights_factor,
            'NUM_LABELS': NUM_LABELS,
            'use_testval': use_testval,
            'TH_TRAIN': TH_TRAIN,
            'TH_VAL': TH_VAL,
            'TH_TEST': TH_TEST,
            'HS_FINAL': hs_final,
            'patience': patience
        }

        with open(os.path.join(output_dir, 'metadata.pickle'), 'wb') as f:
            pickle.dump(metadata, f)

        with open(os.path.join(output_dir, 'metadata.pickle'), 'rb') as f:
            metadata = pickle.load(f)
            print(metadata)

        print("=" * 40 )
        print("=" * 40 )
        print("=" * 40 )
        print("=" * 40 )

    print()
    print("*" * 40)
    print("*" * 40)
    print("*" * 40)
    print("*" * 40)
    print()





