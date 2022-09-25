#!/usr/bin/env python
# coding: utf-8

# # C2AE Architecture
# * X:
#     * (N, d)
# * Y:
#     * (N, m)
# * Z:
#     * (N, l)
# 
# ## Three main components:
# * Fx:
#     * Encodes x into latent space z.
# * Fe:
#     * Encodes y into latent space z.
# * Fd:
#     * Decodes z into label space. 
# 
# ## Loss functions:
# 
# $$L_1 = ||F_x(X) - F_e(Y)||^2 s.t. F_x(X)Fx(X)^T = F_e(Y)F_e(Y)^T = I$$
# $$L_2 = \Gamma(F_e, F_d) = \Sigma_i^N E_i$$
# $$E_i = \frac{1}{|y_i^1||y_i^0|} \Sigma_{p,q \in y_i^1\times y_i^0} e^{F_d(F_e(y_i))^q - F_d(F_e(y_I))^p}$$
# 
# ## Combined Loss:
# $$L_1 + \alpha L_2$$

# In[16]:


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import math
import pickle

from C2AE import C2AE, save_model, load_model, Fe, Fx, Fd, eval_metrics, get_predictions

from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader

import os

from textwrap import wrap


from utils import KunischMetrics
from utils import KunischPruner
from utils import DataExplorer
from utils import KunischPlotter


# In[42]:


DSFLAGS = [['base'],
           ['blur']
           #['blur', 'rain', 'ref', 'rot', 'crop', 'elastic'],
           #['blur', 'rain', 'ref', 'rot', 'crop', 'elastic', 'randaug'],
           #['blur', 'rain', 'ref', 'rot', 'elastic'],
           #['crop'],
           #['elastic'],
           #['gausblur'],
           #['mtnblur'],
           #['rain'],
           #['rain', 'ref', 'rot'],
           #['rain', 'ref', 'rot', 'elastic'],
           #['randaug'],
           #['ref'],
           #['ref', 'rot'],
           #['rot']
]

NUMLABELS = [5, 14, 26, 34, 54, 63, 72, 82, 91, 107, 131, 169, 281]
for NUM_LABELS in NUMLABELS:
    for DS_FLAGS in DSFLAGS:

        CROP_TIMES = 1
        RANDOM_TIMES = 1
        ELASTIC_TIMES = 1
        GAUSBLUR_TIMES = 1

        BATCH_SIZE = 100
        PATIENCE = 100
        NUM_EPOCHS = 600
        FEATURES_DIM = 1600

        PATTERNS_AS_FEATURES = True

        # 0 es 3090, 1 y 2 son 2080
        CUDA_ID = 0

        SAVE = True
        K = 4


        # In[43]:


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
        root_dir = '..'

        for i in range(0, K):
            print("Fold ", i)

            exp_name = f"{NUM_LABELS}L"
            print(f"Nombre del experimento: {exp_name}")

            if PATTERNS_AS_FEATURES:
                features_dir = os.path.join(root_dir, 'features', 'patterns', data_flags, f'K{str(i)}')
                labels_dir = os.path.join(root_dir, 'labels', data_flags, str(i))
                output_dir = os.path.join(root_dir, "outputs", "C2AE_images", data_flags, exp_name, str(i))
                model_dir = os.path.join(root_dir, 'models', 'C2AE_images', data_flags, str(i))
                model_path = os.path.join(model_dir, exp_name + '.pth')

            else:
                features_dir = os.path.join(root_dir, 'features', 'alexnet', data_flags, f'K{str(i)}')
                labels_dir = os.path.join(root_dir, 'labels', data_flags, str(i))
                output_dir = os.path.join(root_dir, "outputs", "C2AE_alexnet", data_flags, exp_name, str(i))
                model_dir = os.path.join(root_dir, 'models', 'C2AE_alexnet', data_flags, str(i))
                model_path = os.path.join(model_dir, exp_name + '.pth')


            Kfolds[i] = {
                'labels_dir': labels_dir,
                'output_dir': output_dir,
                'model_path': model_path,
                'features_dir': features_dir,
            }

            if not (os.path.isdir(features_dir) and os.path.isdir(labels_dir)):
                print(features_dir)
                print(labels_dir)
                raise FileNotFoundError("""
                No existen directorios de datos para el conjunto de flags seleccionado. 
                Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation.
                """)

            print("--Feature set encontrado en {}".format(features_dir))
            print("--Labels set encontrado en {}".format(labels_dir))
            print("")


            if SAVE:
                os.makedirs(output_dir, exist_ok = True)
                os.makedirs(model_dir, exist_ok = True)
                print(f"Los resultados se guardarán en: {output_dir}")
                print(f"Los modelos se guardarán en: {model_dir}")


        # In[44]:


        device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')
        print(f"Usando device: {torch.cuda.get_device_name(device)}")


        # ### Training

        # In[45]:



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

        for i in range(0, K):
            fold = Kfolds[i]
            labels_dir = fold['labels_dir']
            output_dir = fold['output_dir']
            model_path = fold['model_path']
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

            pruner = KunischPruner(NUM_LABELS)
            pruner.set_top_labels(top_labels)

            device = torch.device('cuda')

            train_labels = pd.read_json(os.path.join(labels_dir, 'augmented_train_df.json'), orient='index')
            train_labels = pruner.filter_df(train_labels)

            train_x = pd.read_json(os.path.join(features_dir, 'augmented_train_df.json'), orient='index').values
            train_y = train_labels.values

            test_labels = pd.read_json(os.path.join(labels_dir, 'test_df.json'), orient='index')
            test_labels = pruner.filter_df(test_labels)

            test_x = pd.read_json(os.path.join(features_dir, 'test_df.json'), orient='index').values
            test_y = test_labels.values

            train_dataset = TensorDataset(torch.tensor(train_x,
                                                       device=device,
                                                       dtype=torch.float),
                                          torch.tensor(train_y,
                                                       device=device,
                                                       dtype=torch.float))
            test_dataset = TensorDataset(torch.tensor(test_x,
                                                      device=device,
                                                      dtype=torch.float),
                                         torch.tensor(test_y,
                                                      device=device,
                                                      dtype=torch.float))


            # Training configs.
            num_epochs = NUM_EPOCHS
            batch_size = BATCH_SIZE
            lr = 0.0001
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # # Scene config
            feat_dim = FEATURES_DIM
            latent_dim = 70
            num_labels = NUM_LABELS
            fx_h_dim= 120
            fe_h_dim= 120
            fd_h_dim= 120

            # Scene models.
            Fx_scene = Fx(feat_dim, fx_h_dim, fx_h_dim, latent_dim)
            Fe_scene = Fe(num_labels, fe_h_dim, latent_dim)
            Fd_scene = Fd(latent_dim, fd_h_dim, num_labels, fin_act=torch.sigmoid)

            # Initializing net.
            net = C2AE(Fx_scene, Fe_scene, Fd_scene, beta=0.5, alpha=10, emb_lambda=0.01, latent_dim=latent_dim, device=device)
            net = net.to(device)


            # Doing weight_decay here is eqiv to adding the L2 norm.
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

            print("Starting training!")
            best_weights = None
            best_loss = 9999999
            patience = PATIENCE
            bad_epochs = 0

            for epoch in range(num_epochs+1):
                # Training.
                net.train()
                loss_tracker = 0.0
                latent_loss_tracker = 0.0
                cor_loss_tracker = 0.0
                for x, y in train_dataloader:
                    optimizer.zero_grad()

                    # Pass x, y to network. Retrieve both encodings, and decoding of ys encoding.
                    fx_x, fe_y, fd_z = net(x, y)
                    # Calc loss.
                    l_loss, c_loss = net.losses(fx_x, fe_y, fd_z, y)
                    # Normalize losses by batch.
                    l_loss /= x.shape[0]
                    c_loss /= x.shape[0]
                    loss = net.beta*l_loss + net.alpha*c_loss
                    loss.backward()
                    optimizer.step()

                    loss_tracker+=loss.item()
                    latent_loss_tracker+=l_loss.item()
                    cor_loss_tracker+=c_loss.item()

                # Evaluation
                net.eval()
                loss_tracker = 0.0
                latent_loss_tracker = 0.0
                cor_loss_tracker = 0.0
                acc_track = 0.0
                for x, y in test_dataloader:
                    # evaluation only requires x. As its just Fd(Fx(x))
                    fx_x, fe_y = net.Fx(x), net.Fe(y)
                    fd_z = net.Fd(fx_x)

                    l_loss, c_loss = net.losses(fx_x, fe_y, fd_z, y)
                    # Normalize losses by batch.
                    l_loss /= x.shape[0]
                    c_loss /= x.shape[0]
                    loss = net.beta*l_loss + net.alpha*c_loss

                    latent_loss_tracker += l_loss.item()
                    cor_loss_tracker += c_loss.item()
                    loss_tracker += loss.item()
                    lab_preds = torch.round(net.Fd(net.Fx(x))).cpu().detach().numpy()

                print(f"Epoch: {epoch}, Loss: {loss_tracker},  L-Loss: {latent_loss_tracker}, C-Loss: {cor_loss_tracker}")
                if cor_loss_tracker < best_loss:
                    best_loss = cor_loss_tracker
                    best_weights = net.state_dict()
                    bad_epochs = 0
                    best_epoch = epoch
                else:
                    bad_epochs += 1
                    if bad_epochs == patience:
                        print(f"Out of patience at epoch {epoch}")
                        break

            if SAVE:
                print("Guardando mejor modelo en ", model_path)
                torch.save(best_weights, model_path)

            eval_net = load_model(C2AE, model_path,
                                  Fx=Fx_scene, Fe=Fe_scene, Fd=Fd_scene, device=device).to(device)

            y_pred, y_true = get_predictions(net, [test_dataset], device)

            metrics = KunischMetrics(y_true, y_pred)
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

            print(f"HS fold {i}: {metrics.hs()}")

            if SAVE:
                save_df = pd.DataFrame(y_pred)
                save_df.to_csv(os.path.join(output_dir, 'predictions.csv'))
                print(f"Predicciones guardadas en {os.path.join(output_dir, 'predictions.csv')}")


        avg_f1 = round(sum_f1/K, 4)
        avg_recall = round(sum_recall/K, 4)
        avg_precision = round(sum_precision/K, 4)
        avg_acc = round(sum_acc/K, 4)
        avg_hl = round(sum_hl/K, 4)
        avg_emr = round(sum_emr/K, 4)
        avg_hs = round(sum_hs/K, 4)
        avg_mr1 = round(sum_mr1/K, 4)
        avg_mr2 = round(sum_mr2/K, 4)
        avg_mr3 = round(sum_mr3/K, 4)
        avg_mr4 = round(sum_mr4/K, 4)
        avg_mr5 = round(sum_mr5/K, 4)

        metadata = {
        'data_flags': data_flags,
        'patience': PATIENCE,
        'batch_size': BATCH_SIZE,
        'optimizer': (type (optimizer).__name__),
        'epochs': num_epochs,
        'num_labels': NUM_LABELS,
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

        print("HS Final: ", avg_hs)
        print("F1 Final: ", avg_f1)
        print("1MR Final: ", avg_mr1)
        print("5MR Final: ", avg_mr5)

        if SAVE:
            metadf = pd.DataFrame.from_dict(metadata, orient='index')
            # output_dir pero sin numero de fold
            if PATTERNS_AS_FEATURES:
                os.makedirs(os.path.join(root_dir, 'outputs', 'C2AE_images', data_flags, exp_name), exist_ok=True)
                metadf.to_csv(os.path.join(root_dir, "outputs", "C2AE_images", data_flags, exp_name, 'metadata.csv'))
            else:
                os.makedirs(os.path.join(root_dir, 'outputs', 'C2AE_alexnet', data_flags, exp_name), exist_ok=True)
                metadf.to_csv(os.path.join(root_dir, "outputs", "C2AE_alexnet", data_flags, exp_name, 'metadata.csv'))


        # In[ ]:





        # In[ ]:




