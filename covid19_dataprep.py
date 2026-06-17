#!/usr/bin/env python3

import os
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
                           , interpolation = interp_method) # join(img_dir, filename) pour ça soit compatible toutes OS
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

def display_img_np_arr(img_arr):
    '''
    Display the type, dimensions and minimum and maximum values of a numpy array
    '''
    print(type(img_arr))
    print(img_arr.dtype)
    print(img_arr.shape)
    print(f"Value range: [{img_arr.min()}, {img_arr.max()}]")   

def get_pixel_stats(class_image_arrays):
    '''
    This function returns stats about the pixels once stored in a numpy array
    '''
    
    stats = []

    for class_name, arr in class_image_arrays.items():
        stats.append({
            "class": class_name,
            "n_images": arr.shape[0],
            "pixel_min": arr.min(),
            "pixel_max": arr.max(),
            "pixel_mean": arr.mean(),
            "pixel_std": arr.std()
        })

    return pd.DataFrame(stats)

def get_image_level_pixel_stats(class_image_arrays, dark_threshold=30, bright_threshold=240):

    '''
    This function returns stats about the pixel levels once stored in a numpy array
    '''
    
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