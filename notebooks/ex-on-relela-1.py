#!/usr/bin/env python
# coding: utf-8

# # Aprendizaje Multietiqueta de Patrones Geométricos en Objetos de Herencia Cultural
# # Split and data augmentation
# ## Seminario de Tesis II, Primavera 2022
# ### Master of Data Science. Universidad de Chile.
# #### Prof. guía: Benjamín Bustos - Prof. coguía: Iván Sipirán
# #### Autor: Matías Vergara
# 
# Performs data augmentation on patterns through the application of linear transformations.

# ## Imports

# In[1]:


import cv2
import pandas as pd
from IPython.display import display
import os
import math
import random
import shutil
import imgaug.augmenters as aug
import numpy as np


# ## Mounting Google Drive

# In[2]:


# Mounting google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    root_dir = 'drive/MyDrive/TesisMV/'
except:
    root_dir = '../'


# ## Dataset and model selection

# It is enough to select ds flags and number of crops (this one will have effect depending on flags) and then run the rest of the cells. This way, two folders will be created: a labels one and a patterns one. Both of them will be named after the selected flags, separed by "_"

# In[3]:

FLAGS = [
    #['ref', 'rot', 'rain', 'elastic'],
         #['ref', 'rot', 'rain', 'elastic', 'blur'],
    ['ref', 'rot', 'rain', 'elastic', 'blur', 'crop'],
    ['ref', 'rot', 'rain', 'elastic', 'blur', 'crop', 'randaug'],

]

labels_dir = os.path.join(root_dir, "labels")
for DS_FLAGS in FLAGS:
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
    MAP_TIMES = {'crop': CROP_TIMES,
             'randaug': RANDOM_TIMES,
             'elastic': ELASTIC_TIMES,
             'gausblur': GAUSBLUR_TIMES,
        }

    # Variables globales
    SUBCHAPTERS = False
    K = 4 # k fold


    # ## Transformations

    # In[4]:


    DS_FLAGS = sorted(DS_FLAGS)
    data_flags = '_'.join(DS_FLAGS) if len(DS_FLAGS) > 0 else 'base'
    if SUBCHAPTERS:
        data_flags = 'subchapters/' + data_flags
    MULTIPLE_TRANSF = ['crop', 'randaug', 'elastic', 'gausblur']
    COPY_FLAGS = DS_FLAGS.copy()

    for t in MULTIPLE_TRANSF:
        if t in DS_FLAGS:
            COPY_FLAGS.remove(t)
            COPY_FLAGS.append(t + str(MAP_TIMES[t]))
            data_flags = '_'.join(COPY_FLAGS)

    print(DS_FLAGS)


    # In[5]:


    def rotate90(path):
        image = cv2.imread(path)
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("90", rotated)
        return rotated, "rot90"


    def rotate180(path):
        image = cv2.imread(path)
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        # cv2.imshow("180", rotated)
        return rotated, "rot180"


    def rotate270(path):
        image = cv2.imread(path)
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        # cv2.imshow("270", rotated)
        return rotated, "rot270"


    def invertX(path):
        image = cv2.imread(path)
        flipped = cv2.flip(image, 1)
        # cv2.imshow("flipX", flipped)
        return flipped, "invX"


    def invertY(path):
        image = cv2.imread(path)
        flipped = cv2.flip(image, 0)
        # cv2.imshow("flipY", flipped)
        return flipped, "invY"


    def crop(path, min_width = 1/2, min_height= 1/2, max_width = 1/1.1,
             max_height = 1/1.1):
        image = cv2.imread(path)
        height, width = image.shape[0], image.shape[1] # Caution: there are images in RGB and GS
        min_width = math.ceil(width * min_width)
        min_height = math.ceil(height * min_height)
        max_width = math.ceil(width * max_width)
        max_height = math.ceil(height * max_height)
        x1 = random.randint(0, width - min_width)
        w = random.randint(min_width, width - x1)
        y1 = random.randint(0, height - min_height)
        h = random.randint(min_height, height - y1)
        crop = image[y1:y1+h, x1:x1+w]
        return crop, "crop"

    def blur(path):
        image = cv2.imread(path)
        image_aug = aug.AverageBlur(k=(4, 11))(image=image)
        return image_aug, "blur"

    def gausblur(path):
        image = cv2.imread(path)
        image_aug = aug.GaussianBlur(sigma=random.uniform(2,10))(image=image)
        return image_aug, "gausblur"

    def msblur(path):
        image = cv2.imread(path)
        image_aug = aug.MeanShiftBlur()(image=image)
        return image_aug, "msblur"

    def mtnblur(path):
        image = cv2.imread(path)
        image_aug = aug.MotionBlur(random.randint(10,359))(image=image)
        return image_aug, "mtnblur"


    def emboss(path):
        image = cv2.imread(path)
        image_aug = aug.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))(image = image)
        return image_aug, "embs"

    def elastic(path):
        image = cv2.imread(path)
        image_aug = aug.PiecewiseAffine(scale=(0.03, 0.075))(image = image)
        return image_aug, "elastic"

    def randaug(path):
        image = cv2.imread(path)
        image_aug = aug.RandAugment(m=(2, 9))(image = image)
        return image_aug, "randaug"

    def snow(path):
        image = cv2.imread(path)
        image_aug = aug.Snowflakes(flake_size=(0.6, 0.5), speed=(0.2, 0.5))(image = image)
        return image_aug, "snow"


    def rain(path):
        image = cv2.imread(path)
        image_aug = aug.Rain(speed=(0.1, 0.5))(image = image)
        return image_aug, "rain"


    def apply_transformations(pin, pout, transformations):
        # ../patterns/originals/84e/84e.png
        new_names = []
        i = 0
        for transformation in transformations:
            result, transf_name = transformation(pin)
            if transf_name in MULTIPLE_TRANSF: # special treatment for crops and randoms
              transf_name += str(i)
              i+=1
            pin = os.path.normpath(pin)
            path_els = pin.split(os.sep)
            obj_name = path_els[3] + "_" + transf_name
            filename = obj_name + ".png"
            os.makedirs(pout, exist_ok = True)
            cv2.imwrite(os.path.join(pout, filename), result)
            new_names.append(obj_name)
        return new_names

    # Select data augmentation functions based on data flags

    MAP_FLAGS = {'ref': [invertX, invertY],
                 'rot': [rotate90, rotate180, rotate270],
                 'crop': [crop],
                 'blur': [blur],
                 'gausblur': [gausblur],
                 'mtnblur': [mtnblur],
                 'msblur': [msblur],
                 'emboss': [emboss],
                 'randaug': [randaug],
                 'rain': [rain],
                 'elastic': [elastic]
                 # snow is not working properly
                 }

    ALLOWED_TRANSFORMATIONS = []
    for f in DS_FLAGS:
        ALLOWED_TRANSFORMATIONS += MAP_FLAGS[f]

    HOR_TRANSFORMATIONS = [invertX, rotate180, blur, rain, emboss, mtnblur, gausblur, msblur]
    VER_TRANSFORMATIONS = [invertY, rotate180, blur, rain, emboss, mtnblur, gausblur, msblur]
    COMMON_TRANSFORMATIONS = [invertX, invertY, rotate90, rotate180, rotate270,
                              blur, rain, emboss, mtnblur, gausblur, msblur]

    for t in MULTIPLE_TRANSF:
        if t in DS_FLAGS:
            HOR_TRANSFORMATIONS += MAP_FLAGS[t] * MAP_TIMES[t]
            VER_TRANSFORMATIONS += MAP_FLAGS[t] * MAP_TIMES[t]
            COMMON_TRANSFORMATIONS += MAP_FLAGS[t] * MAP_TIMES[t]


    def mergeTransformations(flags, map_flags, map_times, trans_list):
        # could be improved a lot
        for k, v in map_flags.items():
            if k not in flags:
                for el in v:
                    while el in trans_list:
                        trans_list.remove(el)
        print(trans_list)
        return trans_list

    mergeTransformations(DS_FLAGS, MAP_FLAGS, MAP_TIMES, HOR_TRANSFORMATIONS)
    mergeTransformations(DS_FLAGS, MAP_FLAGS, MAP_TIMES, VER_TRANSFORMATIONS)
    mergeTransformations(DS_FLAGS, MAP_FLAGS, MAP_TIMES, COMMON_TRANSFORMATIONS)


    # In[6]:


    labels_dir = os.path.join(root_dir, "labels")
    df = pd.read_json(os.path.join(labels_dir, "normalized_df.json"), orient='index')
    classes = pd.read_csv(os.path.join(labels_dir, "class_labels.csv"), index_col=0)
    colnames = df.columns
    holdout_dir = os.path.join(labels_dir, "holdout")
    os.makedirs(holdout_dir, exist_ok = True)

    GENERAR = True
    ERROR = False

    train_sets = []
    val_sets = []
    test_sets = []

    for i in range(0, K):
        found_train_elems = os.path.isfile(os.path.join(holdout_dir, "elem_train_" + str(i) + ".npy"))
        found_val_elems = os.path.isfile(os.path.join(holdout_dir, "elem_val_" + str(i) + ".npy"))
        found_test_elems = os.path.isfile(os.path.join(holdout_dir, "elem_test_" + str(i) + ".npy"))

        if (not found_train_elems) or (not found_val_elems) or (not found_test_elems):
            print("No se encontraron los datos del fold ", i)
            ERROR = True

    if ERROR and not GENERAR:
            raise Exception("""
            No hay particiones para CV pero GENERAR está seteado como False. 
            Revise los paths o cambie el valor de GENERAR a True.
            """)

    if not ERROR: #archivos existian desde antes
        print("Cargando indices previamente generados")
        for i in range(0, K):
            elem_train = elem_test = elem_val = None
            with open(os.path.join(holdout_dir, f'elem_train_{i}.npy'), 'rb') as f:
                elem_train = np.load(f, allow_pickle = True)

            with open(os.path.join(holdout_dir, f'elem_val_{i}.npy'), 'rb') as f:
                elem_val = np.load(f, allow_pickle = True)

            with open(os.path.join(holdout_dir, f'elem_test_{i}.npy'), 'rb') as f:
                elem_test = np.load(f, allow_pickle = True)

            train_sets.append(elem_train)
            val_sets.append(elem_val)
            test_sets.append(elem_test)

            print(f"Elementos del fold {i}")
            print(f"-- Elementos de entrenamiento: {len(elem_train)} - Muestra: {elem_train[0:5]}" )
            print(f"-- Elementos de validación: {len(elem_val)} - Muestra: {elem_val[0:5]}")
            print(f"-- Elementos de test: {len(elem_test)} - Muestra: {elem_test[0:5]}")

    if ERROR and GENERAR:
        print("GENERAR está activado. Generando particiones nuevas")

        df = df.sample(frac=1)
        index = df.index.values

        testNumber = math.ceil(len(index)/K)
        valNumber = math.ceil(0.1 * len(index))
        trainNumber = len(index) - valNumber - testNumber
        print(valNumber, testNumber, trainNumber)

        assert (valNumber + testNumber + trainNumber) == len(index)

        for i in range(0, K):
            first_test_index = i * testNumber
            last_test_index = (i + 1) * testNumber
            print(first_test_index, last_test_index)

            test_set = index[first_test_index : last_test_index]

            resto = np.setdiff1d(index, test_set)
            np.random.shuffle(resto)

            val_set = resto[0 : valNumber]
            resto = np.setdiff1d(resto, val_set)
            np.random.shuffle(resto)

            train_set = resto

            test_sets.append(test_set)
            val_sets.append(val_set)
            train_sets.append(train_set)

            print(f"Fold {i}")
            print("-- Patterns for training: {} - Muestra: {}".format(len(train_set), sorted(train_set[0:5])))
            print("- Patterns for validation: {} - Muestra: {}".format(len(val_set), sorted(val_set[0:5])))
            print("-- Patterns for testing: {} - Muestra: {}".format(len(test_set), sorted(test_set[0:5])))

            with open(os.path.join(holdout_dir, f'elem_train_{i}.npy'), 'wb') as f:
                np.save(f, train_set)

            with open(os.path.join(holdout_dir, f'elem_val_{i}.npy'), 'wb') as f:
                np.save(f, val_set)

            with open(os.path.join(holdout_dir, f'elem_test_{i}.npy'), 'wb') as f:
                np.save(f, test_set)

        # Chequear que la K-Cross-validation está sin overlap en test
        for i, set1 in enumerate(test_sets):
            for j, set2 in enumerate(test_sets):
                if i != j:
                    assert len( np.intersect1d(set1, set2) ) == 0



    #
    # ## Augmentation
    # (Only over training set)

    # In[7]:


    for i in range(0, K):

        new_entries = {}
        train_set = train_sets[i]
        val_set = val_sets[i]
        test_set = test_sets[i]

        for pattern in train_set: # only training set
            labels = df.loc[[pattern]]
            lbl_class = classes.loc[[pattern]]['chapter'].values[0]

            if SUBCHAPTERS:
                lbl_class = classes.loc[[pattern]]['subchapter'].values[0]

            path_in = os.path.join(root_dir, "patterns", "originals", pattern, pattern + ".png")
            path_out = os.path.join(root_dir, "patterns", data_flags, str(i), "train", lbl_class)
            is_hor = labels['horizontal'].values[0]
            is_ver = labels['vertical'].values[0]

            if is_hor and is_ver:
                pass

            if is_hor and not is_ver:
                new_names = apply_transformations(path_in, path_out, HOR_TRANSFORMATIONS)
                labels = df.loc[[pattern]].values[0]

            elif is_ver and not is_hor:
                new_names = apply_transformations(path_in, path_out, VER_TRANSFORMATIONS)
                labels = df.loc[[pattern]].values[0]

            else: #if not is_hor and not is_ver:
                new_names = apply_transformations(path_in, path_out, COMMON_TRANSFORMATIONS)
                labels = df.loc[[pattern]].values[0]

            for name in new_names:
                new_entries[name] = labels

            # add the base pattern to the folder
            os.makedirs(path_out, exist_ok = True)
            shutil.copy(path_in, path_out)

        for pattern in val_set:
            lbl_class = classes.loc[[pattern]]['chapter'].values[0]
            if SUBCHAPTERS:
                lbl_class = classes.loc[[pattern]]['subchapter'].values[0]
            path_in = os.path.join(root_dir, "patterns", "originals", pattern, pattern + ".png")
            path_out = os.path.join(root_dir, "patterns", data_flags, str(i), "val", lbl_class)
            os.makedirs(path_out, exist_ok = True)
            shutil.copy(path_in, path_out)

        for pattern in test_set:
            lbl_class = classes.loc[[pattern]]['chapter'].values[0]
            if SUBCHAPTERS:
                lbl_class = classes.loc[[pattern]]['subchapter'].values[0]
            path_in = os.path.join(root_dir, "patterns", "originals", pattern, pattern + ".png")
            path_out = os.path.join(root_dir, "patterns", data_flags, str(i), "test", lbl_class)
            os.makedirs(path_out, exist_ok = True)
            shutil.copy(path_in, path_out)

        # agregar todas las entradas de train a new_entries, y crear
        # el dataset "augmented_train_df.json"

        for p in train_set:
          labels = df.loc[p]
          new_entries[p] = labels.values

        labels_output = os.path.join(labels_dir, data_flags, str(i))

        os.makedirs(labels_output, exist_ok = True)

        df_train = pd.DataFrame.from_dict(new_entries, columns=colnames, orient='index')
        df_train.to_json(os.path.join(labels_output, "augmented_train_df.json"), orient='index')

        # agregar todas las entradas de val a val_entries, y crear
        # el dataset "val_df.json"
        val_entries = {}
        for p in val_set:
          labels = df.loc[p]
          val_entries[p] = labels.values

        df_val = pd.DataFrame.from_dict(val_entries, columns=colnames, orient='index')
        df_val.to_json(os.path.join(labels_output, "val_df.json"), orient='index')

        # agregar todas las entradas de test a test_entries, y crear
        # el dataset "test_df.json"
        test_entries = {}
        for p in test_set:
          labels = df.loc[p]
          test_entries[p] = labels.values

        df_test = pd.DataFrame.from_dict(test_entries, columns=colnames, orient='index')
        df_test.to_json(os.path.join(labels_output, "test_df.json"), orient='index')


    # In[8]:


    # assert(df_train.shape[0] + df_test.shape[0] + df_val.shape[0] == 776)

