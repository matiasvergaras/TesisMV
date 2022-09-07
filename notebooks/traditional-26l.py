#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # Testing multilabel algorithms
# ## Seminario de Tesis II, Primavera 2022
# ### Master of Data Science. Universidad de Chile.
# #### Prof. guía: Benjamín Bustos - Prof. coguía: Iván Sipirán
# #### Autor: Matías Vergara
# 
# ### Referencias:
# 
# Zhang, M. L., Li, Y. K., Liu, X. Y., & Geng, X. (2018). Binary relevance for multi-label learning: an overview. Frontiers of Computer Science, 12(2), 191–202.
# https://doi.org/10.1007/s11704-017-7031-7
# 
# Kariuki C. Multi-Label Classification with Scikit-MultiLearn. 
# https://www.section.io/engineering-education/multi-label-classification-with-scikit-multilearn/

# ## Montando Google Drive
# 

# In[13]:



root_dir = '..'
    
import os


# ## Selección de dataset y modelos

# In[26]:


DSFLAGS = [#[],
           #['blur'],
           #['blur', 'rain', 'ref', 'rot', 'crop', 'elastic'],
           #['blur', 'rain', 'ref', 'rot', 'crop', 'elastic', 'randaug'],
           #['blur', 'rain', 'ref', 'rot', 'elastic'],
           ['crop'],
           ['elastic'],
           ['gausblur'],
           ['mtnblur'],
           ['rain'],
           ['rain', 'ref', 'rot'],
           ['rain', 'ref', 'rot', 'elastic'],
           ['randaug'],
           ['ref'],
           ['ref', 'rot'],
           ['rot']
           ]
USERN50 = [True]#, True]

for USE_RN50 in USERN50:
    for DS_FLAGS in DSFLAGS:
        SUBCHAPTERS = False
        ARCHITECTURE = 'resnet'
        CROP_TIMES = 1
        RANDOM_TIMES = 1
        ELASTIC_TIMES = 1
        GAUSBLUR_TIMES = 1
        NUM_LABELS = 26
        K = 4


        # In[52]:


        # This cells builds the data_flags variable, that will be used
        # to map the requested data treatment to folders
        MAP_TIMES = {'crop': CROP_TIMES,
                 'randaug': RANDOM_TIMES,
                 'elastic': ELASTIC_TIMES,
                 'gausblur': GAUSBLUR_TIMES
        }

        DS_FLAGS = sorted(DS_FLAGS)
        data_flags = '_'.join(DS_FLAGS) if len(DS_FLAGS) > 0 else 'base'
        MULTIPLE_TRANSF = MAP_TIMES.keys()
        COPY_FLAGS = DS_FLAGS.copy()

        for t in MULTIPLE_TRANSF:
            if t in DS_FLAGS:
                COPY_FLAGS.remove(t)
                COPY_FLAGS.append(t + str(MAP_TIMES[t]))
                data_flags = '_'.join(COPY_FLAGS)

        subchapter_str = 'subchapters' if SUBCHAPTERS else ''
        Kfolds = {}

        for i in range(0, K):
            print("Fold ", i)
            if ARCHITECTURE == 'resnet':
                patterns_dir = os.path.join(root_dir, 'patterns', subchapter_str + data_flags, str(i))
                labels_dir = os.path.join(root_dir, 'labels', subchapter_str + data_flags, str(i))
                exp_name = f'resnet50_K{i}' if USE_RN50 else f'resnet18_K{i}'
                rn = 50 if USE_RN50 else 18
                features_dir = os.path.join(root_dir, 'features', ARCHITECTURE, subchapter_str + data_flags, exp_name)

            else:
                patterns_dir = os.path.join(root_dir, 'patterns', subchapter_str + data_flags, str(i))
                labels_dir = os.path.join(root_dir, 'labels', subchapter_str + data_flags, str(i))
                exp_name = f'K{i}'
                features_dir = os.path.join(root_dir, 'features', ARCHITECTURE, subchapter_str + data_flags, exp_name)

            if not (os.path.isdir(patterns_dir) and os.path.isdir(labels_dir)):
                print(patterns_dir)
                print(labels_dir)
                raise FileNotFoundError("""
                No existen directorios de datos para el conjunto de flags seleccionado. 
                Verifique que el dataset exista y, de lo contrario, llame a Split and Augmentation.
                """)
            if not (os.path.isdir(features_dir)):
                print(features_dir)
                raise FileNotFoundError(f"""
                No se encontraron features para el conjunto de flags seleccionado. 
                Verifique que existan y, de lo contrario, llame a Feature Extraction
                """)

            Kfolds[i] = {
                'patterns_dir': patterns_dir,
                'labels_dir': labels_dir,
                'features_dir': features_dir
            }

            print("--Pattern set encontrado en {}".format(patterns_dir))
            print("--Labels set encontrado en {}".format(labels_dir))
            print("--Features set encontrado en {}".format(features_dir))
            print("")

        #../features/resnet/resnet50_base/
        #../labels/base/
        #Features set encontrado en ../features/resnet/resnet50_base/
        #Labels set encontrado en ../labels/base/


        # In[53]:


        train_filename = "augmented_train_df.json"
        val_filename = "val_df.json"
        test_filename = "test_df.json"


        # # Imports

        # In[54]:


        def warn(*args, **kwargs):
            pass
        import warnings
        warnings.warn = warn

        import joblib
        import sys
        sys.modules['sklearn.externals.joblib'] = joblib

        # Data treatment
        import pandas as pd
        import numpy as np
        from scipy import sparse
        from sklearn.model_selection import train_test_split
        import pickle

        # Base classifiers
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        from sklearn.metrics import accuracy_score,hamming_loss, accuracy_score, f1_score, precision_score, recall_score
        from sklearn.linear_model import LogisticRegression
        from sklearn import svm
        from sklearn import tree

        # Multilabel classifiers - Problem Transformation
        from skmultilearn.problem_transform import BinaryRelevance
        from skmultilearn.problem_transform import ClassifierChain
        from skmultilearn.problem_transform import LabelPowerset
        from skmultilearn.ensemble import RakelD

        # Multilabel classifiers - Algorithm Adaptation
        from skmultilearn.adapt import BRkNNaClassifier
        from skmultilearn.adapt import MLkNN
        from skmultilearn.adapt import MLTSVM

        # Metrics
        from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
        from sklearn.metrics import classification_report

        # Plotting
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Embedding classifiers
        #from skmultilearn.embedding import OpenNetworkEmbedder, CLEMS, SKLearnEmbedder
        #from sklearn.manifold import SpectralEmbedding
        #from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
        #from skmultilearn.embedding import EmbeddingClassifier
        #from sklearn.ensemble import RandomForestRegressor

        from utils import KunischMetrics
        from utils import KunischPruner
        from utils import DataExplorer
        from utils import KunischPlotter


        # # Procesamiento

        # In[55]:


        print("Iniciando revisión de datasets")
        for i in range(0, K):
            fold = Kfolds[i]
            print("Fold ", i)
            features_dir = fold['features_dir']
            labels_dir = fold['labels_dir']

            features_train = pd.read_json(os.path.join(features_dir, train_filename), orient='index')
            features_val = pd.read_json(os.path.join(features_dir, val_filename), orient='index')
            features_test = pd.read_json(os.path.join(features_dir, test_filename), orient='index')

            assert features_train.shape[1] == features_test.shape[1] == features_val.shape[1]
            assert features_test.shape[0] == 194
            assert features_val.shape[0] == 78
            print(f"--Utilizando descriptores de dimensión {features_train.shape[1]}")

            labels_train = pd.read_json(os.path.join(labels_dir, train_filename), orient='index')
            labels_val = pd.read_json(os.path.join(labels_dir, val_filename), orient='index')
            labels_test = pd.read_json(os.path.join(labels_dir, test_filename), orient='index')

            assert labels_train.shape[1] == labels_test.shape[1] == labels_val.shape[1]
            #assert labels_test.shape[0] == 194
            #assert labels_val.shape[0] == 78
            print(f"--Los archivos de etiquetas contienen {labels_train.shape[1]} etiquetas distintas.")
            print("Fold aprobado.")
            print()


        # # Data exploration

        # In[56]:


        k = 0 # cambiar para explorar otros folds
        print(f"Explorando fold {k}")
        fold = Kfolds[k]
        labels_dir = fold['labels_dir']
        labels_train = pd.read_json(os.path.join(labels_dir, train_filename), orient='index')
        labels_val = pd.read_json(os.path.join(labels_dir, val_filename), orient='index')
        labels_test = pd.read_json(os.path.join(labels_dir, test_filename), orient='index')

        train_columns = labels_train.columns.values
        labels_75f = labels_train.loc[['75f']].values
        print('Etiquetas 75f: ' + ''.join(np.where(labels_75f == 1, train_columns, '')[0]))

        explorer = DataExplorer(labels_train, labels_val, labels_test)
        combinations = explorer.get_unique_combinations(study='all')
        metrics = explorer.get_label_metrics(study='all')


        # # Funciones auxiliares

        # In[114]:


        def build_model(mlb_estimator, xtrain, ytrain, xtest, ytest, model=None):
            """Builds a multilabel estimator and runs it over a given train and test data,
               with an optional base classifier model.

            Parameters:
            mlb_estimator (mlb classifier): a PROBLEM_TRANSFORMATION or ALGORITHM_ADAPTATION
                                            method from sklearn-multilabel
            xtrain, ytrain, xtest, ytest (np arrays): train and test data
            model (Base Estimator): optional, ignored if mlb_estimator is part of
                                    ALGORITHM_ADAPTATION methods. Base classifier to be
                                    used with the PROBLEM_TRANSFORMATION methods.

            Returns:
            (dict, np.array): dict with metrics (exact match, hamming loss and score)
                              and array of predictions.
            """
            xtrain = sparse.csr_matrix(xtrain)
            ytrain = sparse.csr_matrix(ytrain)
            xtest = sparse.csr_matrix(xtest)
            ytest = sparse.csr_matrix(ytest)
            if model:
              clf = mlb_estimator(model)
            else:
              clf = mlb_estimator
            clf.fit(xtrain, ytrain)
            clf_predictions = clf.predict(xtest)
            return clf_predictions


        # # Benchmark
        #

        # In[115]:


        TRANSF_METHODS = {"BR": BinaryRelevance, "LP": LabelPowerset,
                          "CC": ClassifierChain, "RakelD": RakelD}
        mlknn = MLkNN(k=1, s=1)
        mltsvm = MLTSVM(c_k=4)
        brknna = BRkNNaClassifier(k=1)
        ADAPT_METHODS = {"BRkNN": brknna, "MLkNN": mlknn, "MLTSVM": mltsvm}
        BASE_CLASSIFIERS = {"LR": LogisticRegression(solver='lbfgs'), "SVC": svm.SVC(),
                            "DT": tree.DecisionTreeClassifier(), "GNB": GaussianNB()}


        # In[117]:


        #LABELS_IN_STUDY = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        #            26, 30, 34, 38, 40, 50, 60, 80, 100]
        LABELS_IN_STUDY = [26]
        #for i in range(26, 300, 10):
        #    LABELS_IN_STUDY.append(i)

        exp_exact_match = {}
        exp_hscore = {}
        exp_hloss = {}
        exp_f1 = {}
        exp_precision = {}
        exp_recall = {}
        exp_acc = {}
        exp_1mr = {}
        exp_2mr = {}
        exp_3mr = {}
        exp_4mr = {}
        exp_5mr = {}

        for meth_name in TRANSF_METHODS.keys():
          for base_name in BASE_CLASSIFIERS.keys():
            exp_exact_match[meth_name + "_" + base_name] = []
            exp_hscore[meth_name + "_" + base_name] = []
            exp_hloss[meth_name + "_" + base_name] = []
            exp_f1[meth_name + "_" + base_name] = []
            exp_precision[meth_name + "_" + base_name] = []
            exp_recall[meth_name + "_" + base_name] = []
            exp_acc[meth_name + "_" + base_name] = []
            exp_1mr[meth_name + "_" + base_name] = []
            exp_2mr[meth_name + "_" + base_name] = []
            exp_3mr[meth_name + "_" + base_name] = []
            exp_4mr[meth_name + "_" + base_name] = []
            exp_5mr[meth_name + "_" + base_name] = []

        for meth_name in ADAPT_METHODS.keys():
          exp_exact_match[meth_name] = []
          exp_hscore[meth_name] = []
          exp_hloss[meth_name] = []
          exp_f1[meth_name] = []
          exp_precision[meth_name] = []
          exp_recall[meth_name] = []
          exp_acc[meth_name] = []
          exp_1mr[meth_name] = []
          exp_2mr[meth_name] = []
          exp_3mr[meth_name] = []
          exp_4mr[meth_name] = []
          exp_5mr[meth_name] = []

        PREVIOUS_LABELS = 0
        USED_FREQS = []

        output_dir = os.path.join(root_dir, 'outputs', ARCHITECTURE,
                                   subchapter_str + data_flags, f'{min(LABELS_IN_STUDY)}-{max(LABELS_IN_STUDY)}L')

        os.makedirs(output_dir, exist_ok = True)

        for i in LABELS_IN_STUDY:
          pruner = KunischPruner(i)
          print("Comenzando con i={}".format(i))

          # Carga o generacion de top labels
          top_labels = None
          if not os.path.isfile(os.path.join(root_dir, 'labels', f'top_{i}L.pickle')):
                save = input(f"Se creará un archivo nuevo para {i} labels con el fold 0. Desea continuar? (y/n)")
                if save == "y":
                    labels_dir = Kfolds[0]['labels_dir']
                    train_labels = pd.read_json(os.path.join(labels_dir, 'augmented_train_df.json'), orient='index')
                    top_labels = pruner.filter_labels(train_labels)
                    with open(os.path.join(root_dir, 'labels', f'top_{i}L.pickle'), 'wb') as f:
                        pickle.dump(top_labels, f)
                    print("Top labels creado con éxito")
                else:
                    raise Exception("No se logró cargar top_labels")
          else:
                print(f"Usando top_labels previamente generados para {i} labels")
                with open(os.path.join(root_dir, 'labels', f'top_{i}L.pickle'), 'rb') as f:
                    top_labels = pickle.load(f)
                print(f"top labels previamente generado contiene {len(top_labels)} etiquetas")

          pruner.set_top_labels(top_labels)

          if len(top_labels) == PREVIOUS_LABELS:
                print(f"Al intentar usar {i} labels, se repitió el valor previo {PREVIOUS_LABELS}. Saltando iteración.")
                continue

          PREVIOUS_LABELS = len(top_labels)
          USED_FREQS.append(i)


          for meth_name, method in ADAPT_METHODS.items():
                print("-Probando suerte con", meth_name)

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

                # Iterate over folds
                for j in range(0, K):
                    #print(f"--Trabajando fold {j}")
                    fold = Kfolds[j]
                    features_dir = fold['features_dir']
                    labels_dir = fold['labels_dir']

                    features_train = pd.read_json(os.path.join(features_dir, train_filename), orient='index')
                    features_val = pd.read_json(os.path.join(features_dir, val_filename), orient='index')
                    features_test = pd.read_json(os.path.join(features_dir, test_filename), orient='index')

                    labels_train = pd.read_json(os.path.join(labels_dir, train_filename), orient='index')
                    labels_val = pd.read_json(os.path.join(labels_dir, val_filename), orient='index')
                    labels_test = pd.read_json(os.path.join(labels_dir, test_filename), orient='index')

                    features_test = pd.concat([features_val, features_test])
                    labels_test = pd.concat([labels_val, labels_test])

                    # Dataset creation

                    X_train = features_train.sort_index()
                    X_test = features_test.sort_index()

                    Y_train = pruner.filter_df(labels_train) # reduce labels to most freq
                    Y_test = pruner.filter_df(labels_test) # in both train and test

                    Y_train = Y_train.sort_index()
                    Y_test = Y_test.sort_index()

                    assert X_train.index.all() == Y_train.index.all()
                    assert X_test.index.all() == Y_test.index.all()

                    predictions_i = build_model(method, X_train, Y_train, X_test, Y_test)
                    metrics = KunischMetrics(Y_test.values, predictions_i)

                    micro_f1 = metrics.f1(average='micro')
                    micro_recall = metrics.recall(average='micro')
                    micro_precision = metrics.precision(average='micro')
                    acc = metrics.acc()
                    hl = metrics.hl()
                    emr = metrics.emr()
                    hs = metrics.hs()

                    mr1 = metrics.mr1()
                    mr2 = metrics.mr2()
                    mr3 = metrics.mr3()
                    mr4 = metrics.mr4()
                    mr5 = metrics.mr5()

                    #print("---Micro F1:", micro_f1)
                    #print("---Micro recall:", micro_recall)
                    #print("---Micro precision:", micro_precision)
                    #print("---Accuracy:", acc)
                    #print("---Hamming Loss:", hl)
                    #print("---Exact Match Ratio:", emr)
                    #print("---Hamming Score:", hs)
                    #print("---5-Match Ratio:", mr5)

                    sum_f1 += micro_f1
                    sum_recall += micro_recall
                    sum_precision += micro_precision
                    sum_acc += acc
                    sum_hl += hl
                    sum_emr += emr
                    sum_hs += hs

                    sum_mr1 += mr1
                    sum_mr2 += mr2
                    sum_mr3 += mr3
                    sum_mr4 += mr4
                    sum_mr5 += mr5
                    #print("")

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

                print("---AVG Micro F1:", avg_f1)
                print("---AVG Micro recall:", avg_recall)
                print("---AVG Micro precision:", avg_precision)
                print("---AVG Accuracy:", avg_acc)
                print("---AVG Hamming Loss:", avg_hl)
                print("---AVG Exact Match Ratio:", avg_emr)
                print("---AVG Hamming Score:", avg_hs)
                print("---AVG 1-Match Ratio:", avg_mr1)
                print("---AVG 2-Match Ratio:", avg_mr2)
                print("---AVG 3-Match Ratio:", avg_mr3)
                print("---AVG 4-Match Ratio:", avg_mr4)
                print("---AVG 5-Match Ratio:", avg_mr5)

                exp_exact_match[meth_name].append(avg_emr)
                exp_hscore[meth_name].append(avg_hs)
                exp_hloss[meth_name].append(avg_hl)
                exp_precision[meth_name].append(avg_precision)
                exp_recall[meth_name].append(avg_recall)
                exp_f1[meth_name].append(avg_f1)
                exp_acc[meth_name].append(avg_acc)

                exp_1mr[meth_name].append(avg_mr1)
                exp_2mr[meth_name].append(avg_mr2)
                exp_3mr[meth_name].append(avg_mr3)
                exp_4mr[meth_name].append(avg_mr4)
                exp_5mr[meth_name].append(avg_mr5)

                print("")

          # Linear regression and SVC will raise error if Y_train is composed by only one class
          for meth_name, method in TRANSF_METHODS.items():
                for base_name, classifier in BASE_CLASSIFIERS.items():
                    print("-Probando suerte con", meth_name, base_name)

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

                    # Iterate over folds
                    for j in range(0, K):
                        #print(f"--Trabajando fold {i}")
                        fold = Kfolds[j]
                        features_dir = fold['features_dir']
                        labels_dir = fold['labels_dir']
                        features_train = pd.read_json(os.path.join(features_dir, train_filename), orient='index')
                        features_val = pd.read_json(os.path.join(features_dir, val_filename), orient='index')
                        features_test = pd.read_json(os.path.join(features_dir, test_filename), orient='index')
                        labels_train = pd.read_json(os.path.join(labels_dir, train_filename), orient='index')
                        labels_val = pd.read_json(os.path.join(labels_dir, val_filename), orient='index')
                        labels_test = pd.read_json(os.path.join(labels_dir, test_filename), orient='index')

                        # Dataset creation
                        X_train = features_train.sort_index()
                        X_test = features_test.sort_index()

                        Y_train = pruner.filter_df(labels_train) # reduce labels to most freq
                        Y_test = pruner.filter_df(labels_test) # in both train and test

                        Y_train = Y_train.sort_index()
                        Y_test = Y_test.sort_index()

                        assert X_train.index.all() == Y_train.index.all()
                        assert X_test.index.all() == Y_test.index.all()

                        predictions_i = build_model(method, X_train, Y_train, X_test, Y_test, model=classifier)
                        metrics = KunischMetrics(Y_test.values, predictions_i)

                        micro_f1 = metrics.f1(average='micro')
                        micro_recall = metrics.recall(average='micro')
                        micro_precision = metrics.precision(average='micro')
                        acc = metrics.acc()
                        hl = metrics.hl()
                        emr = metrics.emr()
                        hs = metrics.hs()

                        mr1 = metrics.mr1()
                        mr2 = metrics.mr2()
                        mr3 = metrics.mr3()
                        mr4 = metrics.mr4()
                        mr5 = metrics.mr5()
                        #print("---Micro F1:", micro_f1)
                        #print("---Micro recall:", micro_recall)
                        #print("---Micro precision:", micro_precision)
                        #print("---Accuracy:", acc)
                        #print("---Hamming Loss:", hl)
                        #print("---Exact Match Ratio:", emr)
                        #print("---Hamming Score:", hs)
                        #print("---5-Match Ratio:", mr5)

                        sum_f1 += micro_f1
                        sum_recall += micro_recall
                        sum_precision += micro_precision
                        sum_acc += acc
                        sum_hl += hl
                        sum_emr += emr
                        sum_hs += hs

                        sum_mr1 += mr1
                        sum_mr2 += mr2
                        sum_mr3 += mr3
                        sum_mr4 += mr4
                        sum_mr5 += mr5
                        #print("")

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

                    print("---AVG Micro F1:", avg_f1)
                    print("---AVG Micro recall:", avg_recall)
                    print("---AVG Micro precision:", avg_precision)
                    print("---AVG Accuracy:", avg_acc)
                    print("---AVG Hamming Loss:", avg_hl)
                    print("---AVG Exact Match Ratio:", avg_emr)
                    print("---AVG Hamming Score:", avg_hs)
                    print("---AVG 1-Match Ratio:", avg_mr1)
                    print("---AVG 2-Match Ratio:", avg_mr2)
                    print("---AVG 3-Match Ratio:", avg_mr3)
                    print("---AVG 4-Match Ratio:", avg_mr4)
                    print("---AVG 5-Match Ratio:", avg_mr5)

                    exp_exact_match[meth_name + "_" + base_name].append(avg_emr)
                    exp_hscore[meth_name + "_" + base_name].append(avg_hs)
                    exp_hloss[meth_name + "_" + base_name].append(avg_hl)
                    exp_precision[meth_name + "_" + base_name].append(avg_precision)
                    exp_recall[meth_name + "_" + base_name].append(avg_recall)
                    exp_f1[meth_name + "_" + base_name].append(avg_f1)
                    exp_acc[meth_name + "_" + base_name].append(avg_acc)
                    exp_1mr[meth_name + "_" + base_name].append(avg_mr1)
                    exp_2mr[meth_name + "_" + base_name].append(avg_mr2)
                    exp_3mr[meth_name + "_" + base_name].append(avg_mr3)
                    exp_4mr[meth_name + "_" + base_name].append(avg_mr4)
                    exp_5mr[meth_name + "_" + base_name].append(avg_mr5)
                    print("")


        # In[120]:


        df_output = {}
        for meth_name in exp_exact_match.keys():
            meth_results = {
                'accuracy': exp_acc[meth_name][0],
                'hamming_score': exp_hscore[meth_name][0],
                'hamming_loss': exp_hloss[meth_name][0],
                'f1_score': exp_f1[meth_name][0],
                'recall': exp_recall[meth_name][0],
                'precision': exp_precision[meth_name][0],
                'emr': exp_exact_match[meth_name][0],
                '1mr': exp_1mr[meth_name][0],
                '2mr': exp_2mr[meth_name][0],
                '3mr': exp_3mr[meth_name][0],
                '4mr': exp_4mr[meth_name][0],
                '5mr': exp_5mr[meth_name][0],
            }
            df_output[meth_name] = meth_results

        df_output = pd.DataFrame.from_dict(df_output, orient='index')
        os.makedirs(output_dir, exist_ok=True)
        df_output.to_csv(os.path.join(output_dir, 'resultados.csv'))


