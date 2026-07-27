import os
import numpy as np
import pandas as pd
import covid19_dataprep as prep
import covid19_dataviz as viz
import pickle

workspace = 'data/COVID-19_Radiography_Dataset/'

def load_covid():
    covid_img_dir = os.path.join(workspace+'COVID/images/')
    covid_img_arr = prep.load_img_as_np_arr(covid_img_dir)
    covid_msk_dir = os.path.join(workspace+'COVID/masks/')
    covid_msk_arr = prep.load_img_as_np_arr(covid_msk_dir)
    unique_covid_img_arr, unique_covid_msk_arr = prep.unique_np_arr(covid_img_arr, covid_msk_arr)
    return covid_img_arr, covid_msk_arr, unique_covid_img_arr, unique_covid_msk_arr

covid_img_arr, covid_msk_arr, unique_covid_img_arr, unique_covid_msk_arr = load_covid()

def load_lung_opa():
    lung_opa_img_dir = os.path.join(workspace+'Lung_Opacity/images/')
    lung_opa_img_arr = prep.load_img_as_np_arr(lung_opa_img_dir)
    lung_opa_msk_dir = os.path.join(workspace+'Lung_Opacity/masks/')
    lung_opa_msk_arr = prep.load_img_as_np_arr(lung_opa_msk_dir)
    unique_lung_opa_img_arr, unique_lung_opa_msk_arr = prep.unique_np_arr(lung_opa_img_arr, lung_opa_msk_arr)
    return lung_opa_img_arr, lung_opa_msk_arr, unique_lung_opa_img_arr, unique_lung_opa_msk_arr

lung_opa_img_arr, lung_opa_msk_arr, unique_lung_opa_img_arr, unique_lung_opa_msk_arr = load_lung_opa()

def load_normal():
    normal_img_dir = os.path.join(workspace+'Normal/images/')
    normal_img_arr = prep.load_img_as_np_arr(normal_img_dir)
    normal_msk_dir = os.path.join(workspace+'Normal/masks/')
    normal_msk_arr = prep.load_img_as_np_arr(normal_msk_dir)
    unique_normal_img_arr, unique_normal_msk_arr = prep.unique_np_arr(normal_img_arr, normal_msk_arr)
    return normal_img_arr, normal_msk_arr, unique_normal_img_arr, unique_normal_msk_arr

normal_img_arr, normal_msk_arr, unique_normal_img_arr, unique_normal_msk_arr = load_normal()

def load_pneumo():
    pneumo_img_dir = os.path.join(workspace+'Viral Pneumonia/images/')
    pneumo_img_arr = prep.load_img_as_np_arr(pneumo_img_dir)
    pneumo_msk_dir = os.path.join(workspace+'Viral Pneumonia/masks/')
    pneumo_msk_arr = prep.load_img_as_np_arr(pneumo_msk_dir)
    unique_pneumo_img_arr, unique_pneumo_msk_arr = prep.unique_np_arr(pneumo_img_arr, pneumo_msk_arr)
    return pneumo_img_arr, pneumo_msk_arr, unique_pneumo_img_arr, unique_pneumo_msk_arr

pneumo_img_arr, pneumo_msk_arr, unique_pneumo_img_arr, unique_pneumo_msk_arr = load_pneumo()

dict_to_df = {
"category" : ['COVID'
              , 'Lung_Opacity'
              , 'Normal'
              , 'Viral Pneumonia']
, "number_img" : [covid_img_arr.shape[0]                      
                  , lung_opa_img_arr.shape[0]
                  , normal_img_arr.shape[0]
                  , pneumo_img_arr.shape[0]]
, "unique_number_img" : [unique_covid_img_arr.shape[0]
                  , unique_lung_opa_img_arr.shape[0]
                  , unique_normal_img_arr.shape[0]
                  , unique_pneumo_img_arr.shape[0]]
, "width" : [covid_img_arr.shape[1]
                  , lung_opa_img_arr.shape[1]
                  , normal_img_arr.shape[1]
                  , pneumo_img_arr.shape[1]]
, "height" : [covid_img_arr.shape[2]
                  , lung_opa_img_arr.shape[2]
                  , normal_img_arr.shape[2]
                  , pneumo_img_arr.shape[2]]
}

class_image_arrays = {
    "COVID": unique_covid_img_arr,
    "Lung Opacity": unique_lung_opa_img_arr,
    "Normal": unique_normal_img_arr,
    "Viral Pneumonia": unique_pneumo_img_arr
}

class_mask_arrays = {
    "COVID": unique_covid_msk_arr,
    "Lung Opacity": unique_lung_opa_msk_arr,
    "Normal": unique_normal_msk_arr,
    "Viral Pneumonia": unique_pneumo_msk_arr
}

with open('dict_to_df.pickle', 'wb') as handle:
    pickle.dump(dict_to_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('class_image_arrays.pickle', 'wb') as handle:
    pickle.dump(class_image_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('class_mask_arrays.pickle', 'wb') as handle:
    pickle.dump(class_mask_arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)