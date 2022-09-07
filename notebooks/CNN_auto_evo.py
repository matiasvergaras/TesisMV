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

# In[4]:


# Mounting google Drive
try:
    from google.colab import drive

    drive.mount('/content/drive')
    root_dir = '../content/gdrive/MyDrive'
except:
    root_dir = '..'

# In[5]:


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

# In[6]:

NLABELS = [82, 91, 107, 131, 169, 281]
#NLABELS = [131, 169, 281]
USEPW = [True]
DSFLAGS = [
    ['rain', 'ref', 'rot'],
]
bbb = """
for N in NLABELS:
    pruner = KunischPruner(N)
    if not os.path.isfile(os.path.join(root_dir, 'labels', f'top_{N}L.pickle')):
        print(f"Creando top_labels para {N} labels")
        train_labels = pd.read_json(os.path.join(root_dir, 'labels', 'base', '0', 'augmented_train_df.json'), orient='index')
        top_labels = pruner.filter_labels(train_labels)
        pruner.set_top_labels(top_labels)#
        save = True
        if save:
            with open(os.path.join(root_dir, 'labels', f'top_{N}L.pickle'), 'wb') as f:
                pickle.dump(top_labels, f)
            print("Top labels creado con éxito")
        else:
            raise Exception("No se logró cargar top_labels")

    else:
       print(f"Usando top_labels previamente generados para {N} labels")
       with open(os.path.join(root_dir, 'labels', f'top_{N}L.pickle'), 'rb') as f:
           top_labels = pickle.load(f)
    print("relacion N y top_labels")
    print(N, len(top_labels))
"""

# Evo base, blur con y sin pw
for use_pos_weights in USEPW:
    for N in NLABELS:
        print("Trabajando con {} Labels".format(N))
        for DS_FLAGS in DSFLAGS:
            #           DS_FLAGS = [] cuando este terminando un ciclo cortar por esta razon
            # 'ref': [invertX, invertY],
            # 'rot': [rotate90, rotate180, rotate270],
            # 'crop': [crop] * CROP_TIMES,
            # 'blur': [blur],
            # 'gausblur': [gausblur]
            # 'msblur': [msblur]
            # 'mtnblur': [mtnblur]
            # 'emboss': [emboss],
            # 'randaug': [randaug],
            # 'rain': [rain],
            # 'elastic': [elastic]
            CROP_TIMES = 1
            RANDOM_TIMES = 1
            ELASTIC_TIMES = 1
            GAUSBLUR_TIMES = 1

            pos_weights_factor = 1
            BATCH_SIZE = 124

            TH_TRAIN = 0.5
            TH_VAL = 0.5
            TH_TEST = 0.5

            # 0 es 3090, 1 y 2 son 2080
            CUDA_ID = 0

            SAVE = True
            K = 4

            # In[9]:

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

            for fold_i in range(0, K):
                print("Fold ", fold_i)
                patterns_dir = os.path.join(root_dir, 'patterns', data_flags, str(fold_i))
                labels_dir = os.path.join(root_dir, 'labels', data_flags, str(fold_i))

                if not (os.path.isdir(patterns_dir) and os.path.isdir(labels_dir)):
                    print(patterns_dir)
                    print(labels_dir)
                    raise FileNotFoundError("""
                    No existen directorios de datos para el conjunto de flags seleccionado. 
                    Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation.
                    """)

                exp_name = f"{N}L"
                weights_str = str(pos_weights_factor)
                weights_str = weights_str.replace('.', '_')
                exp_name += f'_weighted_{weights_str}' if use_pos_weights else ''
                print(f"Nombre del experimento: {exp_name}")

                output_dir = os.path.join(root_dir, "outputs", "alexnet", data_flags, exp_name, str(fold_i))
                model_dir = os.path.join(root_dir, "models", "alexnet", data_flags, str(fold_i))
                model_path = os.path.join(model_dir, exp_name + '.pth')

                Kfolds[fold_i] = {
                    'patterns_dir': patterns_dir,
                    'labels_dir': labels_dir,
                    'output_dir': output_dir,
                    'model_path': model_path
                }

                print("--Pattern set encontrado en {}".format(patterns_dir))
                print("--Labels set encontrado en {}".format(labels_dir))

                if SAVE:
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(model_dir, exist_ok=True)
                    print(f"Los resultados se guardarán en: {output_dir}")
                    print(f"Los modelos se guardarán en: {model_dir}")


            # ## Configuración de dispositivo

            # In[10]:

            device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')
            print(f"Usando device: {torch.cuda.get_device_name(device)}")


            # ## Funciones auxiliares

            # In[11]:

            def make_positive_weights(labels, factor=1):
                total = labels.values.sum()
                weights = [0.] * len(labels)
                for i, label in enumerate(labels):
                    weights[i] = total / (factor * labels[i])
                return weights


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
                    # print(img_id, img_name, self.labels_frame.iloc[idx], self.labels_frame.iloc[idx].values, labels)
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
                        tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
                    # print('tmp_a: {0}'.format(tmp_a))
                    acc_list.append(tmp_a)
                return round(np.mean(acc_list), 4)


            # Define the function for training, validation, and test
            def alex_train(epoch, num_epochs, train_losses, learning_rate, w):
                alex_net.train()
                train_loss = 0
                TN = 0
                TP = 0
                FP = 0
                FN = 0
                preds_total = np.empty((1, N), dtype=int)
                labels_total = np.empty((1, N), dtype=int)

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
                train_losses.append(
                    [epoch, learning_rate, w, train_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])

                # print statistics
                print(
                    'Train Trial [{}/{}], LR: {:.4g}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
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
                preds_total = np.empty((1, N), dtype=int)
                labels_total = np.empty((1, N), dtype=int)
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

                    valid_loss = valid_loss / len(
                        kunischValidationLoader.dataset) * BATCH_SIZE  # 1024 is the batch size
                    valid_losses.append(
                        [epoch, learning_rate, w, valid_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])
                    # print statistics
                    print(
                        'Valid Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
                        .format(epoch, num_epochs, optimizer.param_groups[0]['lr'], w, valid_loss, accuracy, f1_score,
                                hs))
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
                preds_total = np.empty((1, N), dtype=int)
                labels_total = np.empty((1, N), dtype=int)

                with torch.no_grad():
                    for i, sample_batched in enumerate(kunischTestLoader, 1):
                        #print("CURRENT BATCH SIZE: ", BATCH_SIZE)
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

                        for j in range(0, min(BATCH_SIZE, show_images)):  # j itera sobre ejemplos
                            print(f"Mostrando imagen {j} del batch {i}")
                            img = np.transpose(sample_batched['image'][j])  # imagen j
                            plt.imshow(img, interpolation='nearest')
                            plt.show()
                            labels_correctos = ""
                            labels_predichos = ""
                            for k in range(0, len(pred[j])):
                                labels_correctos += (kunischTestSet.labels_frame.columns.values[k] + ' ') if \
                                labels[j].cpu().detach()[k] else ""
                                labels_predichos += (kunischTestSet.labels_frame.columns.values[k] + ' ') if \
                                pred[j].cpu().detach()[k] else ""
                            print("Labels correctos:")
                            # print(labels[j].cpu().detach().numpy())
                            print(labels_correctos)
                            print("Labels predichos:")
                            # print(pred[j].cpu().detach().numpy())
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
                    test_losses.append(
                        [epoch, learning_rate, w, test_loss, TP, TN, FP, FN, accuracy, precision, recall, f1_score])
                    # print statistics
                    print(
                        'Test Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}, HS: {:.4f}'
                        .format(epoch, num_epochs, optimizer.param_groups[0]['lr'], w, test_loss, accuracy, f1_score,
                                hs))
                    return hs


            # ## Experimentos

            # In[12]:

            pruner = KunischPruner(N)

            sum_f1 = 0
            sum_recall = 0
            sum_precision = 0
            sum_acc = 0
            sum_hl = 0
            sum_emr = 0
            sum_hs = 0
            sum_mr1 = 0
            sum_mr2 = 0
            sum_mr3 = 0
            sum_mr4 = 0
            sum_mr5 = 0

            for fold_i in range(0, K):
                fold = Kfolds[fold_i]
                labels_dir = fold['labels_dir']
                patterns_dir = fold['patterns_dir']
                output_dir = fold['output_dir']
                model_path = fold['model_path']
                # Carga de top labels
                train_labels = pd.read_json(os.path.join(labels_dir, 'augmented_train_df.json'), orient='index')

                if not os.path.isfile(os.path.join(root_dir, 'labels', f'top_{N}L.pickle')):
                    print(f"Creando top_labels para {N} labels")
                    top_labels = pruner.filter_labels(train_labels)
                    pruner.set_top_labels(top_labels)

                    save = True
                    if save:
                        with open(os.path.join(root_dir, 'labels', f'top_{N}L.pickle'), 'wb') as f:
                            pickle.dump(top_labels, f)
                        print("Top labels creado con éxito")

                    else:
                        raise Exception("No se logró cargar top_labels")

                else:
                    print(f"Usando top_labels previamente generados para {N} labels")
                    with open(os.path.join(root_dir, 'labels', f'top_{N}L.pickle'), 'rb') as f:
                        top_labels = pickle.load(f)

                print("relacion N y top_labels")
                print(N, len(top_labels))
                #assert N == len(top_labels)

                # Creacion de pesos positivos
                if use_pos_weights:
                    pos_weights = make_positive_weights(top_labels, pos_weights_factor)
                    pos_weights = torch.Tensor(pos_weights).float().to(device)
                    print("Pesos positivos creados")
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

                kunischTrainLoader = torch.utils.data.DataLoader(kunischTrainSet, batch_size=BATCH_SIZE, shuffle=True,
                                                                 num_workers=0)

                # Validation
                kunischValidationSet = KunischDataset(images_dir=os.path.join(patterns_dir, 'val'),
                                                      labels_file=os.path.join(labels_dir, 'val_df.json'),
                                                      transform=transforms.Compose([transforms.Resize((227, 227)),
                                                                                    transforms.ToTensor(),
                                                                                    transforms.Normalize(
                                                                                        mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])]),
                                                      top_labels=top_labels)

                kunischValidationLoader = torch.utils.data.DataLoader(kunischValidationSet, batch_size=BATCH_SIZE,
                                                                      shuffle=True,
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

                kunischTestLoader = torch.utils.data.DataLoader(kunischTestSet, batch_size=BATCH_SIZE, shuffle=False,
                                                                num_workers=0)

                hyperval = """
                # Hyper Parameter Tuning
                alex_net = models.alexnet(pretrained=True)
                for param in alex_net.parameters():
                    param.requires_grad = False
                alex_net.classifier._modules['6'] = nn.Linear(4096, NUM_LABELS)
            
                train_losses = []
                validation_losses = []
                num_epochs = 5
            
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
            
                  alex_train(epoch, num_epochs, train_losses, learning_rate, w)
                  if SAVE:
                      train_losses_df = pd.DataFrame(train_losses)
                      train_losses_df.to_csv(os.path.join(output_dir, 'loss_hypertrain.csv'))
            
                  alex_valid(epoch, num_epochs, validation_losses, learning_rate, w)
                  if SAVE:
                      validation_losses_df = pd.DataFrame(validation_losses)
                      validation_losses_df.to_csv(os.path.join(output_dir, 'loss_hyperval.csv'))
                 """

                # Training
                from torch.optim.lr_scheduler import StepLR

                train_losses = []
                validation_losses = []
                num_epochs = 200
                learning_rate = 0.001
                w = 0.01

                # Early Stopping
                patience = 15
                bad_epochs = 0
                best_score = 0.0
                best_weights = None

                alex_net = models.alexnet(pretrained=True)
                alex_net.classifier._modules['6'] = nn.Linear(4096, N)
                alex_net.to(device)

                optimizer = torch.optim.Adam(alex_net.parameters(), lr=learning_rate, weight_decay=w)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=5,
                                                                       min_lr=0.0001)
                # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
                print("Comenzando entrenamiento")
                for epoch in range(num_epochs):
                    print("Epoch ", epoch)
                    score_train = alex_train(epoch, num_epochs, train_losses, learning_rate, w)
                    score_valid = alex_valid(epoch, num_epochs, validation_losses, learning_rate, w)

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

                if SAVE:
                    print(f"Guardando mejor modelo en {model_path}")
                    torch.save(best_weights, model_path)

                # alerta sonora para cuando haya terminado de entrenar
                # import winsound
                # import time
                # winsound.Beep(freq, 2000)

                # Testing
                test_losses = []
                test_pred = []
                learning_rate = 0.0001
                w = 0.001

                # Reset Model
                alex_net = models.alexnet(pretrained=True)
                alex_net.classifier._modules['6'] = nn.Linear(4096, N)
                alex_net.load_state_dict(torch.load(model_path))
                alex_net.to(device)

                optimizer = torch.optim.Adam(alex_net.parameters(), lr=learning_rate, weight_decay=w)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
                alex_test(1, 1, test_pred, test_losses, learning_rate, w, show_images=0)
                test_pred_df = pd.DataFrame(test_pred)

                # Guardar resultados
                preds = test_pred[0][3].cpu().detach().numpy()
                for pred_i in range(1, len(test_pred)):
                    pbi = test_pred[pred_i][3].cpu().detach().numpy()
                    preds = np.concatenate((preds, pbi), axis=0)

                if SAVE:
                    save_df = pd.DataFrame(preds)
                    save_df.to_csv(os.path.join(output_dir, 'predictions.csv'))
                    print(f"Predicciones guardadas en {os.path.join(output_dir, 'predictions.csv')}")
                    preds = pd.read_csv(os.path.join(output_dir, 'predictions.csv'), index_col=0)
                    preds = preds.values

                pruner = KunischPruner(preds.shape[1])
                pruner.set_top_labels(top_labels)
                labels_test = pd.read_json(os.path.join(labels_dir, 'test_df.json'), orient='index')
                test = pruner.filter_df(labels_test)

                metrics = KunischMetrics(test.values, preds)
                sum_f1 += metrics.f1()
                sum_recall += metrics.recall()
                sum_precision += metrics.precision()
                sum_acc += metrics.acc()
                sum_hl += metrics.hl()
                sum_emr += metrics.emr()
                sum_hs += metrics.hs()
                sum_mr1 += metrics.mr1()
                sum_mr2 += metrics.mr2()
                sum_mr3 += metrics.mr3()
                sum_mr4 += metrics.mr4()
                sum_mr5 += metrics.mr5()

                print(f"HS fold {fold_i}: {metrics.hs()}")

            avg_f1 = round(sum_f1 / K, 4)
            avg_recall = round(sum_recall / K, 4)
            avg_precision = round(sum_precision / K, 4)
            avg_acc = round(sum_acc / K, 4)
            avg_hl = round(sum_hl / K, 4)
            avg_emr = round(sum_emr / K, 4)
            avg_hs = round(sum_hs / K, 4)
            avg_mr1 = round(sum_mr1 / K, 4)
            avg_mr2 = round(sum_mr2 / K, 4)
            avg_mr3 = round(sum_mr3 / K, 4)
            avg_mr4 = round(sum_mr4 / K, 4)
            avg_mr5 = round(sum_mr5 / K, 4)

            metadata = {
                'data_flags': data_flags,
                'use_pos_weights': use_pos_weights,
                'pos_weights_factor': pos_weights_factor,
                'patience': patience,
                'batch_size': BATCH_SIZE,
                'optimizer': (type(optimizer).__name__),
                'scheduler': (type(scheduler).__name__),
                'criterion': (type(criterion).__name__),
                'epochs': num_epochs,
                'best_epoch': best_epoch,
                'num_labels': N,
                'TH_TRAIN': TH_TRAIN,
                'TH_VAL': TH_VAL,
                'TH_TEST': TH_TEST,
                'f1': avg_f1,
                'recall': avg_recall,
                'precision': avg_precision,
                'acc': avg_acc,
                'hl': avg_hl,
                'emr': avg_emr,
                'hs': avg_hs,
                'mr1': avg_mr1,
                'mr2': avg_mr2,
                'mr3': avg_mr3,
                'mr4': avg_mr4,
                'mr5': avg_mr5
            }

            #print("HS Final: ", avg_hs)
            #print("F1 Final: ", avg_f1)
            #print("1MR Final: ", avg_mr1)
            #print("5MR Final: ", avg_mr5)

            if SAVE:
                metadf = pd.DataFrame.from_dict(metadata, orient='index')
                # output_dir pero sin numero de fold
                metadf.to_csv(os.path.join(root_dir, "outputs", "alexnet", data_flags, exp_name, 'metadata.csv'))

            # In[ ]:
