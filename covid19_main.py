import os
import numpy as np
import pandas as pd
import covid19_dataprep as prep
import time




workspace = 'data/COVID-19_Radiography_Dataset/'

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

# class
covid_class_array = np.array([0 for i in range(unique_covid_img_arr.shape[0])])

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

# class
lung_opa_class_array = np.array([1 for i in range(unique_lung_opa_img_arr.shape[0])])

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

# class
normal_class_array = np.array([2 for i in range(unique_normal_img_arr.shape[0])])

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

# class
pneumo_class_array = np.array([3 for i in range(unique_pneumo_img_arr.shape[0])])

end = time.time()
print('Fin :', round(end - start,3), 's')



##################################################
# Array concatenation
##################################################

start = time.time()
print('Début concaténation arrays...')

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

class_array = np.concatenate([covid_class_array
                            , lung_opa_class_array
                            , normal_class_array
                            , pneumo_class_array]
                            , axis = 0)

end = time.time()
print('Fin :', round(end - start,3), 's')


