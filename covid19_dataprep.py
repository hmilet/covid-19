#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

def load_img_as_np_arr(img_dir, new_size = (256,256), interp_method = 'lanczos'):
    '''
    This function reads an image and turns it into a numpy array
    It allows resizing ; for downscaling, the interpolation method 'lanczos' should be used to minimize information loss ; for upscaling, "nearest" should be used to avoid inserting non 1 values
    '''

    img_arr_list = []

    for filename in sorted(os.listdir(img_dir)): # j'ai ajouter sorted pour assurer l'ordre
        if filename.endswith(".png"): 
            img = load_img(os.path.join(img_dir, filename)
                           , target_size = new_size
                           , interpolation = interp_method
                           , color_mode = 'grayscale') # join(img_dir, filename) pour ça soit compatible toutes OS
            arr = img_to_array(img)
            img_arr_list.append(arr)
            continue
        else:
            print(f'{filename} is not a png file')
            continue

    return np.array(img_arr_list)

def unique_np_arr(img_arr, msk_arr):
    '''
    This function allows to return the image and mask arrays without the duplicates
    The focus is on identifying the duplicates in images AND THEN applying the same transformation to masks to prevent removing usable masks
    '''
    # the first underscore allows to ignore the first result
    _, idx_uniques = np.unique(img_arr, axis=0, return_index=True)
    idx_uniques = np.sort(idx_uniques)

    # we apply the same transformation to img and masks
    img_clean = img_arr[idx_uniques]
    msk_clean = msk_arr[idx_uniques]

    return img_clean, msk_clean

def get_image_level_pixel_stats_with_ratios(class_image_arrays, dark_threshold=30, bright_threshold=240):
    """
    Calcule les statistiques par image :
    - luminosité moyenne
    - écart-type des pixels
    - ratio de pixels très sombres
    - ratio de pixels très clairs
    """
    
    stats = []

    for class_name, arr in class_image_arrays.items():
        for i in range(arr.shape[0]):
            img = arr[i]
            
            # Conversion simple en niveaux de gris
            img_gray = img.mean(axis=2)
            
            stats.append({
                "class": class_name,
                "image_index": i,
                "pixel_mean": img_gray.mean(),
                "pixel_std": img_gray.std(),
                "ratio_pixels_tres_sombres": (img_gray < dark_threshold).mean(),
                "ratio_pixels_tres_clairs": (img_gray > bright_threshold).mean()
            })

    return pd.DataFrame(stats)

def add_ratio_thresholds_by_class(stats_df, q_ratio=0.99):
    """
    Ajoute les seuils de ratio sombre et clair par classe.
    Les seuils sont calculés avec un quantile.
    """
    
    df = stats_df.copy()
    
    df["dark_ratio_threshold"] = (
        df.groupby("class")["ratio_pixels_tres_sombres"]
        .transform(lambda s: s.quantile(q_ratio))
    )
    
    df["bright_ratio_threshold"] = (
        df.groupby("class")["ratio_pixels_tres_clairs"]
        .transform(lambda s: s.quantile(q_ratio))
    )
    
    return df

def detect_suspicious_images_by_ratio(stats_df, q_ratio=0.99):
    """
    Détecte les images suspectes selon :
    - un ratio élevé de pixels très sombres
    - un ratio élevé de pixels très clairs
    """
    
    df = add_ratio_thresholds_by_class(stats_df, q_ratio=q_ratio)
    
    df["is_dark_suspicious"] = (
        df["ratio_pixels_tres_sombres"] >= df["dark_ratio_threshold"]
    )
    
    df["is_bright_suspicious"] = (
        (df["ratio_pixels_tres_clairs"] >= df["bright_ratio_threshold"]) &
        (df["ratio_pixels_tres_clairs"] > 0)
    )
    
    df["is_suspicious"] = (
        df["is_dark_suspicious"] | df["is_bright_suspicious"]
    )
    
    return df

def get_suspicious_images(stats_suspicious_df):
    """
    Retourne 3 DataFrames :
    - images suspectes sombres
    - images suspectes claires
    - toutes les images suspectes
    """
    
    dark_suspicious_df = (
        stats_suspicious_df[stats_suspicious_df["is_dark_suspicious"]]
        .sort_values("ratio_pixels_tres_sombres", ascending=False)
        .copy()
    )
    
    bright_suspicious_df = (
        stats_suspicious_df[stats_suspicious_df["is_bright_suspicious"]]
        .sort_values("ratio_pixels_tres_clairs", ascending=False)
        .copy()
    )
    
    all_suspicious_df = (
        stats_suspicious_df[stats_suspicious_df["is_suspicious"]]
        .copy()
    )
    
    return dark_suspicious_df, bright_suspicious_df, all_suspicious_df

def clean_class_image_and_mask_arrays(class_image_arrays, class_mask_arrays, to_remove_df):
    """
    Crée une version nettoyée des images et des masks.
    
    Les images à supprimer sont définies dans to_remove_df avec :
    - class
    - image_index
    
    Important :
    On supprime les mêmes indices dans les images et dans les masks.
    """
    
    class_image_arrays_clean = {}
    class_mask_arrays_clean = {}
    
    remove_map = (
        to_remove_df
        .groupby("class")["image_index"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )
    
    for class_name, img_arr in class_image_arrays.items():
        
        mask_arr = class_mask_arrays[class_name]
        
        if img_arr.shape[0] != mask_arr.shape[0]:
            raise ValueError(
                f"Problème pour la classe {class_name} : "
                f"{img_arr.shape[0]} images mais {mask_arr.shape[0]} masks."
            )
        
        indices_to_remove = remove_map.get(class_name, set())
        
        indices_to_keep = [
            i for i in range(img_arr.shape[0])
            if i not in indices_to_remove
        ]
        
        class_image_arrays_clean[class_name] = img_arr[indices_to_keep]
        class_mask_arrays_clean[class_name] = mask_arr[indices_to_keep]
    
    return class_image_arrays_clean, class_mask_arrays_clean


def augmenter_classe_numpy(X_class, y_train, y_train_flat, label_value, target_size, data_augmentation):
    """
    Génère un array NumPy augmenté pour atteindre exactement la taille cible.
    """
    actual_number_images = len(X_class)
    number_to_generate = target_size - actual_number_images
    
    augmented_images = []
    
    # Looping until target size is matched
    while len(augmented_images) < number_to_generate:
        # Random picking
        idx = np.random.randint(0, actual_number_images)
        img = X_class[idx]
        
        img_batch = np.expand_dims(img, axis=0)
        
        # Augmenting / training = True for randomness
        img_aug = data_augmentation(img_batch, training=True)
        
        # Stripping dimensions
        augmented_images.append(img_aug[0].numpy())
        
    # Numpy array conversions
    X_aug = np.array(augmented_images)
    y_aug = np.full((len(X_aug)), label_value) # Crée les labels correspondants
    
    # Concatenate all the images
    X_final = np.concatenate([X_class, X_aug], axis=0)
    y_final = np.concatenate([y_train[y_train_flat == label_value], y_aug], axis=0)
    
    return X_final, y_final