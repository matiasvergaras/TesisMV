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
    root_dir = '../..'


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


from utils import KunischMetrics
from utils import KunischPruner
from utils import DataExplorer
from utils import KunischPlotter


# ## Selección de dataset y experimento

# In[12]:


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

for DS_FLAGS in DSFLAGS:
    print(DS_FLAGS)
    CROP_TIMES = 1
    RANDOM_TIMES = 1
    ELASTIC_TIMES = 1
    GAUSBLUR_TIMES = 1

    use_pos_weights = True
    pos_weights_factor = 1
    NUM_LABELS = 26
    BATCH_SIZE = 1 #importante para la extraccion

    TH_TRAIN = 0.5
    TH_VAL = 0.5
    TH_TEST = 0.5

    # 0 es 3090, 1 y 2 son 2080
    CUDA_ID = 0

    SAVE = True
    K = 4


    # In[34]:


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

    Kfolds = {}

    for i in range(0, K):
        print("Fold ", i)
        patterns_dir = os.path.join(root_dir, 'patterns', data_flags, str(i))
        labels_dir = os.path.join(root_dir, 'labels', data_flags, str(i))

        if not (os.path.isdir(patterns_dir) and os.path.isdir(labels_dir)):
            print(patterns_dir)
            print(labels_dir)
            raise FileNotFoundError("""
            No existen directorios de datos para el conjunto de flags seleccionado. 
            Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation.
            """)

        #exp_name = f"{NUM_LABELS}L"
        weights_str = str(pos_weights_factor)
        weights_str = weights_str.replace('.','_')
        #exp_name += f'_weighted_{weights_str}' if use_pos_weights else ''
        #print(f"Nombre del experimento: {exp_name}")

        #features_dir = os.path.join(root_dir, "features", "alexnet", data_flags, exp_name, str(i))
        features_dir = os.path.join(root_dir, 'features', 'alexnet', data_flags, 'K' + str(i))
        model_dir = os.path.join(root_dir, "models", "alexnet", data_flags, str(i))
        #model_path = os.path.join(model_dir, exp_name + '.pth')

        Kfolds[i] = {
            'patterns_dir': patterns_dir,
            'labels_dir': labels_dir,
            #'output_dir': output_dir,
            #'model_path': model_path,
            'features_dir': features_dir
        }

        print("--Pattern set encontrado en {}".format(patterns_dir))
        print("--Labels set encontrado en {}".format(labels_dir))
        #print("--Path al modelo {}".format(model_path))
        print("")


        if SAVE:
            os.makedirs(features_dir, exist_ok = True)
            print(f"Las features se guardarán en: {features_dir}")


    # ## Configuración de dispositivo

    # In[35]:


    device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {torch.cuda.get_device_name(device)}")


    # ## Funciones auxiliares

    # In[36]:


    def make_positive_weights(labels, factor=1):
        total = labels.values.sum()
        weights = [0.] * len(labels)
        for i, label in enumerate(labels):
          weights[i] = total/(factor * labels[i])
        return weights

    def get_vector(model,layer, dim_embedding, x):

      my_embedding = torch.zeros(dim_embedding)

      def copy_data(m,i,o):
        my_embedding.copy_(o.data.squeeze())

      h = layer.register_forward_hook(copy_data)
      model(x)
      h.remove()

      return my_embedding

    # images_dir=os.path.join(root_dir, 'patterns', data_flags, 'train'),
    # labels_file=os.path.join(root_dir, 'labels', data_flags, 'augmented_train_df.json'),
    class KunischDataset(torch.utils.data.Dataset):

      def __init__(self, images_dir, labels_file, transform, top_labels):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.pruner = KunischPruner(len(top_labels))
        self.pruner.set_top_labels(top_labels)
        labels = pd.read_json(labels_file, orient='index')
        self.labels_frame = self.pruner.filter_df(labels)
        self.num_labels = len(top_labels)
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.transform = transform
        self.flags = data_flags
        self.top_labels = top_labels

      def __len__(self):
        return len(self.labels_frame)

      def __getitem__(self, idx):
        img_id = self.labels_frame.iloc[idx].name + '.png'
        img_name = None
        for chapter in os.listdir(self.images_dir):
          if img_id in os.listdir(os.path.join(self.images_dir, chapter)):
            img_name = os.path.join(self.images_dir, chapter, img_id)
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

    def hamming_score(y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])
            # print('\nset_true: {0}'.format(set_true))
            # print('set_pred: {0}'.format(set_pred))
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) /                     float(len(set_true.union(set_pred)))
            # print('tmp_a: {0}'.format(tmp_a))
            acc_list.append(tmp_a)
        return round(np.mean(acc_list), 4)

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


    # ## Extraction

    # In[37]:


    pruner = KunischPruner(NUM_LABELS)


    for i in range(0, K):
        fold = Kfolds[i]
        labels_dir = fold['labels_dir']
        patterns_dir = fold['patterns_dir']
        #output_dir = fold['output_dir']
        #model_path = fold['model_path']
        features_dir = fold['features_dir']
        # Carga de top labels
        train_labels = pd.read_json(os.path.join(labels_dir, 'augmented_train_df.json'), orient='index')

        if not os.path.isfile(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle')):
            print(f"Creando top_labels para {NUM_LABELS} labels")
            top_labels = pruner.filter_labels(train_labels)
            pruner.set_top_labels(top_labels)

            save = input(f"Se creará un archivo nuevo para {len(top_labels)} labels. Desea continuar? (y/n)")
            if save == "y":
                with open(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle'), 'wb') as f:
                    pickle.dump(top_labels, f)
                print("Top labels creado con éxito")

            else:
                raise Exception("No se logró cargar top_labels")

        else:
            print(f"Usando top_labels previamente generados para {NUM_LABELS} labels")
            with open(os.path.join(root_dir, 'labels', f'top_{NUM_LABELS}L.pickle'), 'rb') as f:
                top_labels = pickle.load(f)

        NUM_LABELS = len(top_labels) # la cantidad final de etiquetas a trabajar

        # Creacion de pesos positivos
        if use_pos_weights:
            pos_weights = make_positive_weights(top_labels, pos_weights_factor)
            pos_weights = torch.Tensor(pos_weights).float().to(device)

        else:
            pos_weights = None

        # Alexnet requires 227 x 227
        # Training
        kunischTrainSet = KunischDataset(images_dir=os.path.join(patterns_dir, 'train'),
                                         labels_file=os.path.join(labels_dir, 'augmented_train_df.json'),
                                         transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(
                                                                           mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])]),
                                         top_labels=top_labels)

        kunischTrainLoader = torch.utils.data.DataLoader(kunischTrainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Validation
        kunischValidationSet = KunischDataset(images_dir=os.path.join(patterns_dir, 'val'),
                                              labels_file=os.path.join(labels_dir, 'val_df.json'),
                                              transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize(
                                                                                mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]),
                                              top_labels=top_labels)

        kunischValidationLoader = torch.utils.data.DataLoader(kunischValidationSet, batch_size=BATCH_SIZE, shuffle=True,
                                                              num_workers=0)

        # Test
        kunischTestSet = KunischDataset(images_dir=os.path.join(patterns_dir, 'test'),
                                        labels_file=os.path.join(labels_dir, 'test_df.json'),
                                        transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                      transforms.ToTensor(),
                                                                      transforms.Normalize(
                                                                          mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])]),
                                        top_labels=top_labels)

        kunischTestLoader = torch.utils.data.DataLoader(kunischTestSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


        alex_net = models.alexnet(pretrained=True)
        for param in alex_net.parameters():
            param.requires_grad = False
        alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)
        #alex_net.load_state_dict(torch.load(model_path))
        alex_net.to(device)

        layer =  alex_net.classifier[4]
        dim = alex_net.classifier[4].in_features

        alex_net.eval()

        #features = []
        features_train = {}
        features_val = {}
        features_test = {}


        for data in kunischTrainLoader:
          img = data['image']
          name = os.path.normpath(data['paths'][0],).split('\\')
          name = name[len(name)-1]
          feat = get_vector(alex_net, layer, dim, img.to(device))
          namefile = name
          code, rest = namefile.split('.')
          features_train[code] = feat.numpy().tolist()
          #features.append(feat.numpy())

        for data in kunischValidationLoader:
          img = data['image']
          name = os.path.normpath(data['paths'][0],).split('\\')
          name = name[len(name)-1]
          feat = get_vector(alex_net, layer, dim, img.to(device))
          namefile = name
          code, rest = namefile.split('.')
          features_val[code] = feat.numpy().tolist()

        for data in kunischTestLoader:
          img = data['image']
          name = os.path.normpath(data['paths'][0],).split('\\')
          name = name[len(name)-1]
          feat = get_vector(alex_net, layer, dim, img.to(device))
          namefile = name
          code, rest = namefile.split('.')
          features_test[code] = feat.numpy().tolist()
        #features = np.vstack(features)

        os.makedirs(features_dir, exist_ok=True)

        features_train_df = pd.DataFrame.from_dict(features_train, orient='index')
        features_val_df = pd.DataFrame.from_dict(features_val, orient='index')
        features_test_df = pd.DataFrame.from_dict(features_test, orient='index')

        output_train = os.path.join(features_dir, 'augmented_train_df.json')
        output_val = os.path.join(features_dir, 'val_df.json')
        output_test = os.path.join(features_dir, 'test_df.json')

        print("Guardando en {}".format(output_train))
        print("Guardando en {}".format(output_val))
        print("Guardando en {}".format(output_test))

        features_train_df.to_json(output_train, orient='index')
        features_val_df.to_json(output_val, orient='index')
        features_test_df.to_json(output_test, orient='index')


    # In[ ]:




