import os
import numpy as np
import pandas as pd
import covid19_dataprep as prep
import time
import argparse
import covid19_svm_utils as svm
import covid19_randomforest_utils as randomforest
import covid19_cnn_utils as cnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score
import tensorflow as tf
from tensorflow.keras import layers
import pickle


##################################################
##################################################
#
# Arguments parsing
#
##################################################
##################################################

parser = argparse.ArgumentParser(description="Covid-19 X-ray classification")

parser.add_argument('-i', action='store_true', help="For a first run to preprocess and save the data on the disk")
parser.add_argument('-o', action='store_true', help="To add the outlier treatment based on quantiles")
parser.add_argument('-d', action='store_true', help="Generate the X train and y train objects and write them as files")
parser.add_argument('-t', action='store_true', help="Allows the model to train ; else will try to load from a file")


args = parser.parse_args()

workspace = 'data/COVID-19_Radiography_Dataset/'

if args.i:
    ##################################################
    ##################################################
    #
    # Data preparation
    #
    ##################################################
    ##################################################


    ##################################################
    # COVID
    ##################################################

    start = time.time()
    print('Début chargement COVID...')

    # img
    covid_img_dir = os.path.join(workspace+'COVID/images/')
    covid_img_arr = prep.load_img_as_np_arr(covid_img_dir)

    # msk
    covid_msk_dir = os.path.join(workspace+'COVID/masks/')
    covid_msk_arr = prep.load_img_as_np_arr(covid_msk_dir)

    # uniques
    unique_covid_img_arr, unique_covid_msk_arr = prep.unique_np_arr(covid_img_arr, covid_msk_arr)

    end = time.time()
    print('Fin :', round(end - start, 3), 's')



    ##################################################
    # Lung Opacity
    ##################################################

    start = time.time()
    print('Début chargement Lung Opacity...')

    # img
    lung_opa_img_dir = os.path.join(workspace+'Lung_Opacity/images/')
    lung_opa_img_arr = prep.load_img_as_np_arr(lung_opa_img_dir)

    # msk
    lung_opa_msk_dir = os.path.join(workspace+'Lung_Opacity/masks/')
    lung_opa_msk_arr = prep.load_img_as_np_arr(lung_opa_msk_dir)

    # uniques
    unique_lung_opa_img_arr, unique_lung_opa_msk_arr = prep.unique_np_arr(lung_opa_img_arr, lung_opa_msk_arr)

    end = time.time()
    print('Fin :', round(end - start,3), 's')



    ##################################################
    # Normal
    ##################################################

    start = time.time()
    print('Début chargement Normal...')

    # img
    normal_img_dir = os.path.join(workspace+'Normal/images/')
    normal_img_arr = prep.load_img_as_np_arr(normal_img_dir)

    # msk
    normal_msk_dir = os.path.join(workspace+'Normal/masks/')
    normal_msk_arr = prep.load_img_as_np_arr(normal_msk_dir)

    # uniques
    unique_normal_img_arr, unique_normal_msk_arr = prep.unique_np_arr(normal_img_arr, normal_msk_arr)

    end = time.time()
    print('Fin :', round(end - start,3), 's')



    ##################################################
    # Viral Pneumonia
    ##################################################

    start = time.time()
    print('Début chargement Pneumonia...')

    # img
    pneumo_img_dir = os.path.join(workspace+'Viral Pneumonia/images/')
    pneumo_img_arr = prep.load_img_as_np_arr(pneumo_img_dir)

    # msk
    pneumo_msk_dir = os.path.join(workspace+'Viral Pneumonia/masks/')
    pneumo_msk_arr = prep.load_img_as_np_arr(pneumo_msk_dir)

    # uniques
    unique_pneumo_img_arr, unique_pneumo_msk_arr = prep.unique_np_arr(pneumo_img_arr, pneumo_msk_arr)

    end = time.time()
    print('Fin :', round(end - start,3), 's')


    if args.o : 
        ##################################################
        # Outliers processing
        ##################################################

        start = time.time()
        print('Début gestion des outliers...')

        class_image_arrays = {
            "COVID": unique_covid_img_arr,
            "Lung Opacity": unique_lung_opa_img_arr,
            "Normal": unique_normal_img_arr,
            "Viral Pneumonia": unique_pneumo_img_arr
        }

        image_level_stats_df = prep.get_image_level_pixel_stats_with_ratios(class_image_arrays, dark_threshold=30, bright_threshold=240)

        # Appliquer la détection
        stats_suspicious_df = prep.detect_suspicious_images_by_ratio(
            image_level_stats_df,
            q_ratio=0.99
        )

        _, _, all_suspicious_df = prep.get_suspicious_images(
            stats_suspicious_df
        )

        # Préparer les images à retirer
        to_remove_df = all_suspicious_df[[
            "class",
            "image_index",
            "pixel_mean",
            "pixel_std",
            "ratio_pixels_tres_sombres",
            "ratio_pixels_tres_clairs",
            "is_dark_suspicious",
            "is_bright_suspicious"
        ]].drop_duplicates().copy()

        class_mask_arrays = {
            "COVID": unique_covid_msk_arr,
            "Lung Opacity": unique_lung_opa_msk_arr,
            "Normal": unique_normal_msk_arr,
            "Viral Pneumonia": unique_pneumo_msk_arr
        }
        # Créer le dataset nettoyé
        class_image_arrays_clean, class_mask_arrays_clean = prep.clean_class_image_and_mask_arrays(
            class_image_arrays,
            class_mask_arrays,
            to_remove_df
        )

        unique_covid_img_arr = class_image_arrays_clean['COVID']
        unique_lung_opa_img_arr = class_image_arrays_clean['Lung Opacity']
        unique_normal_img_arr = class_image_arrays_clean['Normal']
        unique_pneumo_img_arr = class_image_arrays_clean['Viral Pneumonia']

        unique_covid_msk_arr = class_mask_arrays_clean['COVID']
        unique_lung_opa_msk_arr = class_mask_arrays_clean['Lung Opacity']
        unique_normal_msk_arr = class_mask_arrays_clean['Normal']
        unique_pneumo_msk_arr = class_mask_arrays_clean['Viral Pneumonia']

        end = time.time()
        print('Fin :', round(end - start,3), 's')

    ##################################################
    # Class arrays
    ##################################################

    # 0 = COVID
    # 1 = Lung Opacity
    # 2 = Normal
    # 3 = Pneumonia

    covid_class_array = np.array([0 for i in range(unique_covid_img_arr.shape[0])])

    lung_opa_class_array = np.array([1 for i in range(unique_lung_opa_img_arr.shape[0])])

    normal_class_array = np.array([2 for i in range(unique_normal_img_arr.shape[0])])

    pneumo_class_array = np.array([3 for i in range(unique_pneumo_img_arr.shape[0])])

    ##################################################
    # Array concatenation
    ##################################################

    start = time.time()
    print('Début concaténation arrays...')

    # features
    img_array = np.concatenate([unique_covid_img_arr
                                , unique_lung_opa_img_arr
                                , unique_normal_img_arr
                                , unique_pneumo_img_arr]
                                , axis = 0)

    msk_array = np.concatenate([unique_covid_msk_arr
                                , unique_lung_opa_msk_arr
                                , unique_normal_msk_arr
                                , unique_pneumo_msk_arr]
                                , axis = 0)

    msk_array = msk_array / 255

    cropped_img_array = img_array * msk_array

    # target
    class_array = np.concatenate([covid_class_array
                                , lung_opa_class_array
                                , normal_class_array
                                , pneumo_class_array]
                                , axis = 0)

    end = time.time()
    print('Fin :', round(end - start,3), 's')


    ##################################################
    # Save preprocessed arrays
    ##################################################

    start = time.time()
    print('Début écriture fichiers...')

    np.save('img_array.npy', img_array.astype(np.uint8))
    np.save('cropped_img_array.npy', cropped_img_array.astype(np.uint8))
    np.save('class_array.npy', class_array.astype(np.uint8))

    end = time.time()
    print('Fin :', round(end - start,3), 's')

else:
    
    if args.d:

        ##################################################
        # Data augmentation
        ##################################################

        start = time.time()
        print('Début chargement arrays locales...')

        img_array = np.load('img_array.npy')
        cropped_img_array = np.load('cropped_img_array.npy')
        class_array = np.load('class_array.npy')

        end = time.time()
        print('Fin :', round(end - start,3), 's')

        X_train, X_test, y_train, y_test = train_test_split(
            #img_array,# test 1-5
            cropped_img_array,
            class_array,
            test_size=0.2,
            random_state=66,
            stratify=class_array
        )

        start = time.time()
        print("Génération des données augmentées en cours...")

        data_augmentation = tf.keras.Sequential([
            #layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.15),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            layers.RandomContrast(factor=0.1),
        ])

        y_train_flat = y_train.flatten()
        mask_covid = (y_train_flat == 0)
        mask_lung_opa = (y_train_flat == 1)
        mask_normal = (y_train_flat == 2)
        mask_pneumo = (y_train_flat == 3)

        X_covid, y_covid = X_train[mask_covid], y_train[mask_covid]
        X_lung_opa, y_lung_opa = X_train[mask_lung_opa], y_train[mask_lung_opa]
        X_normal, y_normal = X_train[mask_normal], y_train[mask_normal]
        X_pneumo, y_pneumo = X_train[mask_pneumo], y_train[mask_pneumo]

        # Size to match for other classes
        target_size = len(X_normal)

        X_normal_res, y_normal_res = X_normal, y_train[mask_normal] # Inchangée
        X_covid_res, y_covid_res = prep.augmenter_classe_numpy(X_covid, y_train, y_train_flat, 0, target_size, data_augmentation)
        X_lung_opa_res, y_lung_opa_res = prep.augmenter_classe_numpy(X_lung_opa, y_train, y_train_flat, 1, target_size, data_augmentation)
        X_pneumo_res, y_pneumo_res = prep.augmenter_classe_numpy(X_pneumo, y_train, y_train_flat, 3, target_size, data_augmentation)

        X_train_balanced = np.concatenate([X_normal_res, X_covid_res, X_lung_opa_res, X_pneumo_res], axis=0)
        y_train_balanced = np.concatenate([y_normal_res, y_covid_res, y_lung_opa_res, y_pneumo_res], axis=0)

        # Reshuffle of indexes
        indexes = np.arange(len(X_train_balanced))
        np.random.shuffle(indexes)

        X_train = X_train_balanced[indexes]
        y_train = y_train_balanced[indexes]

        with open('xtest.pickle', 'wb') as handle:
            pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('ytest.pickle', 'wb') as handle:
            pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('xtrain.pickle', 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('ytrain.pickle', 'wb') as handle:
            pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print("Fin :", round(end - start, 3), "s")

    if not args.d:
        with open('xtest.pickle', 'rb') as handle:
            X_test = pickle.load(handle)
        with open('ytest.pickle', 'rb') as handle:
            y_test = pickle.load(handle)
        with open('xtrain.pickle', 'rb') as handle:
            X_train = pickle.load(handle)
        with open('ytrain.pickle', 'rb') as handle:
            y_train = pickle.load(handle)

    if args.t :

        ##################################################
        # CNN
        ##################################################

        start = time.time()
        print("Entraînement du CNN...")

        model = cnn.train_cnn_model(
            X_train = X_train,
            y_train = y_train,
            input_shape = (256,256,1), 
            target_size = (256,256,1),
            epochs = 500,
            batch_size = 64,
            patience = 50
        )

        with open('model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print("Fin :", round(end - start, 3), "s")

    if not args.t :

        start = time.time()
        print("Chargement du modèle...")

        with open('model.pickle', 'rb') as handle:
            model = pickle.load(handle)
        with open('xtest.pickle', 'rb') as handle:
            X_test = pickle.load(handle)
        with open('ytest.pickle', 'rb') as handle:
            y_test = pickle.load(handle)

        end = time.time()
        print("Fin :", round(end - start, 3), "s")


    cnn.evaluate_cnn_model(
        model = model,
        X_test = X_test,
        y_test = y_test
    )

    # #### test RandomForestClassifier

    # start = time.time()
    # print('Début PCA + RF...')

    # random_forest_results = randomforest.train_evaluate_pca_randomforest(
    #                         X_train=X_train_balanced,
    #                         y_train=y_train_balanced,
    #                         X_test=X_test,
    #                         y_test=y_test,
    #                         #scoring_mode="covid",
    #                         cv=3,
    #                         n_jobs=-1,
    #                         verbose=2
    #                     )
    
    # print("Best params :", random_forest_results["best_params"])
    # print("Best CV score :", random_forest_results["best_cv_score"])
    # print("Variance expliquée PCA :", random_forest_results["explained_variance_ratio"])

    # print("Accuracy :", random_forest_results["accuracy"])
    # print("Macro precision :", random_forest_results["macro_precision"])
    # print("Macro recall :", random_forest_results["macro_recall"])
    # print("Macro F1 :", random_forest_results["macro_f1"])

    # print(random_forest_results["classification_report"])
    # print(random_forest_results["confusion_matrix"])

    # end = time.time()
    # print("Fin :", round(end - start, 3), "s")

    ##### test SVM
    # pca_svm_results = svm.test_pca_svm(
    #                         X_train=X_train,
    #                         y_train=y_train,
    #                         X_test=X_test,
    #                         y_test=y_test,
    #                         scoring_mode="covid",
    #                         cv=3,
    #                         n_jobs=3,
    #                         verbose=1
    #                     )
    # start = time.time()
    # print("Début test PCA + SVM...")

    # #
    # covid_f1_scorer = make_scorer(
    #     f1_score,
    #     labels=[0],
    #     average="macro",
    #     zero_division=0
    # )

    # # param grid pour le pipeline
    # param_grid = [
    # {
    #     "pca__n_components": [150],
    #     "svm__kernel": ["rbf"],
    #     "svm__C": [0.5, 1, 2, 5],
    #     "svm__gamma": ["scale", 0.0005, 0.001, 0.005]
    # }
    # ]

    # # test PCA + SVM
    # pca_svm_results = svm.train_evaluate_pca_svm(
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test,
    #     param_grid=param_grid,
    #     #scoring="f1_macro",# test 1 avec f1 score macro = avg f1 score de toutes les classes
    #     scoring=covid_f1_scorer,# test 2 avec f1 score pour la classe covid
    #     cv=3,
    #     n_jobs=1,
    #     verbose=1
    # )

    # print("Best params :", pca_svm_results["best_params"])
    # print("Best CV score :", pca_svm_results["best_cv_score"])
    # print("Variance expliquée PCA :", pca_svm_results["explained_variance_ratio"])

    # print("Accuracy :", pca_svm_results["accuracy"])
    # print("Macro precision :", pca_svm_results["macro_precision"])
    # print("Macro recall :", pca_svm_results["macro_recall"])
    # print("Macro F1 :", pca_svm_results["macro_f1"])

    # print(pca_svm_results["classification_report"])
    # print(pca_svm_results["confusion_matrix"])

    # end = time.time()
    # print("Fin :", round(end - start, 3), "s")