import os
import numpy as np
import pandas as pd
import covid19_dataprep as prep
import time
import argparse
import covid19_svm_utils as svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score

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

    start = time.time()
    print('Début chargement arrays locales...')

    img_array = np.load('img_array.npy')
    cropped_img_array = np.load('cropped_img_array.npy')
    class_array = np.load('class_array.npy')

    end = time.time()
    print('Fin :', round(end - start,3), 's')



# train test split

# data augmentation

##### test SVM
# svm_results = svm.train_evaluate_svm(
#     X_train=X_train_augmented,
#     y_train=y_train_augmented,
#     X_test=X_test,
#     y_test=y_test,
#     scoring="f1_macro",
#     cv=5
# )
# print(svm_results["macro_f1"])
# print(svm_results["classification_report"])
# print(svm_results["confusion_matrix"])

# start = time.time()
# print('Début test SVM...')
# #features = img_array.reshape(img_array[100].shape[0], -1)
# # split pour test l'algo malgré que c'est pas equilibré
# X_train_augmented, X_test, y_train_augmented, y_test = train_test_split(img_array, class_array, test_size=0.2, random_state=66)
# svm_results = svm.train_evaluate_svm(
#     X_train=X_train_augmented,
#     y_train=y_train_augmented,
#     X_test=X_test,
#     y_test=y_test,
#     scoring="f1_macro",
#     cv=5
# )
# print(svm_results["macro_f1"])
# print(svm_results["classification_report"])
# print(svm_results["confusion_matrix"])

# end = time.time()
# print('Fin :', round(end - start,3), 's')

start = time.time()
print("Début test PCA + SVM...")

# split juste pour le test PCA + SVM malgré le déséquilibre des classes
X_train, X_test, y_train, y_test = train_test_split(
    #img_array,# test 1-5
    cropped_img_array, #
    class_array,
    test_size=0.2,
    random_state=66,
    stratify=class_array
)

#
covid_f1_scorer = make_scorer(
    f1_score,
    labels=[0],
    average="macro",
    zero_division=0
)

# param grid pour le pipeline
param_grid = [
    {
        "pca__n_components": [200, 250, 300],
        "svm__kernel": ["rbf"],
        "svm__C": [7],
        "svm__gamma": [0.001, 0.003, 0.005]
    }
]

# test PCA + SVM
pca_svm_results = svm.train_evaluate_pca_svm(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    param_grid=param_grid,
    #scoring="f1_macro",# test 1 avec f1 score macro = avg f1 score de toutes les classes
    scoring=covid_f1_scorer,# test 2 avec f1 score pour la classe covid
    cv=3,
    n_jobs=1,
    verbose=1
)

print("Best params :", pca_svm_results["best_params"])
print("Best CV score :", pca_svm_results["best_cv_score"])
print("Variance expliquée PCA :", pca_svm_results["explained_variance_ratio"])

print("Accuracy :", pca_svm_results["accuracy"])
print("Macro precision :", pca_svm_results["macro_precision"])
print("Macro recall :", pca_svm_results["macro_recall"])
print("Macro F1 :", pca_svm_results["macro_f1"])

print(pca_svm_results["classification_report"])
print(pca_svm_results["confusion_matrix"])

end = time.time()
print("Fin :", round(end - start, 3), "s")