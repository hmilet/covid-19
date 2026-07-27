"""
Examples: pour le model on peux choisir 'cnn', 'densenet121', 'efficientnetb2', voici des exemples avec différents résolution d'images (size) pour le model densenet121
### image size 128
# 1) charger le dataset complet enuite nettoyer les outlier -> output: class_array_128.npy, cropped_img_array_128.npy, img_array_128.npy
python covid19_main.py -i -o --size 128
# 2) split dataset nettoyer et data augmentation sur le train -> output: xtrain_128_o.pickle, ytrain_128_o.pickle, xtest_128_o.pickle, ytest_128_o.pickle
python covid19_main.py -d -o --size 128
# 3) train puis evaluation jeux de test -> output: densenet121_s128_clean.keras, predictions_densenet121_s128_clean.csv et affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -t -o --size 128 -n s128_clean  --model densenet121 --metric covid_f1 --batch-size 64
# 4) chargement du model densenet121_s128_clean.keras puis evalution jeux de test, affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -o --size 128 -n s128_clean  --model densenet121

### image size 256
# 1) charger le dataset complet enuite nettoyer les outlier -> output: class_array_256.npy, cropped_img_array_256.npy, img_array_256.npy
python covid19_main.py -i -o --size 256
# 2) split dataset nettoyer et data augmentation sur le train -> output: xtrain_256_o.pickle, ytrain_256_o.pickle, xtest_256_o.pickle, ytest_256_o.pickle
python covid19_main.py -d -o --size 256
# 3) train puis evaluation jeux de test -> output: densenet121_clean4.keras, predictions_densenet121_clean4.csv et affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -t -o --size 256 -n clean4  --model densenet121 --metric covid_f1 --batch-size 64
# 4) chargement du model densenet121_clean4.keras puis evalution jeux de test, affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -o --size 256 -n clean4  --model densenet121

### image size 224
# 1) charger le dataset complet enuite nettoyer les outlier -> output: class_array_224.npy, cropped_img_array_224.npy, img_array_224.npy
python covid19_main.py -i -o --size 224
# 2) split dataset nettoyer et data augmentation sur le train -> output: xtrain_224_o.pickle, ytrain_224_o.pickle, xtest_224_o.pickle, ytest_224_o.pickle
python covid19_main.py -d -o --size 224
# 3) train puis evaluation jeux de test -> output: densenet121_s224_clean2.keras, predictions_densenet121_s224_clean2.csv et affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -t -o --size 224 -n s224_clean2  --model densenet121 --metric covid_f1 --batch-size 64
# 4) chargement du model densenet121_s224_clean2.keras puis evalution jeux de test, affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -o --size 224 -n s224_clean2  --model densenet121

### image size 299
# 1) charger le dataset complet enuite nettoyer les outlier -> output: class_array_299.npy, cropped_img_array_299.npy, img_array_299.npy
python covid19_main.py -i -o --purist --size 299
# 2) split dataset nettoyer et data augmentation sur le train -> output: xtrain_299_o.pickle, ytrain_299_o.pickle, xtest_299_o.pickle, ytest_299_o.pickle
python covid19_main.py -d -o --size 299
# 3) train puis evaluation jeux de test -> output: densenet121_s299_clean1.keras, predictions_densenet121_s299_clean1.csv et affichage Classification report, Matrice de confusion, une image gradcam
python covid19_main.py -t -o --size 299 -n s299_clean1  --model densenet121 --metric covid_f1 --batch-size 32
python covid19_main.py -o --size 299 -n s299_clean1  --model densenet121
"""
import os
import numpy as np
import pandas as pd
import covid19_dataprep as prep
import time
import argparse
import covid19_svm_utils as svm
import covid19_randomforest_utils as randomforest
import covid19_cnn_utils as cnn
import covid19_efficientnet_utils as efficientnet
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import covid19_densenet_utils as densenet
import covid19_efficientnet_utils as efficientnet

NATIVE_IMG_SIZE = 299   # résolution native des PNG images
NATIVE_MSK_SIZE = 256   # résolution native des PNG masques

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
# --- Arguments ---
parser.add_argument('--model', type=str, default='cnn',
                    choices=['cnn', 'densenet121', 'efficientnetb2'],
                    help="Modèle à entraîner/évaluer")
parser.add_argument('-n', type=str, default='run1',
                    help="Nom du run, utilisé pour nommer le fichier du modèle")
parser.add_argument('--metric', type=str, default='macro_f1',
                    choices=['macro_f1', 'covid_f1'],
                    help="Métrique suivie pour l'early stopping (DenseNet)")
parser.add_argument('--size', type=int, default=256,
                    choices=[128, 224, 256, 299],
                    help="Résolution cible des images (carrées)")
parser.add_argument('--purist', action='store_true',
                    help="Charge les PNG directement à --size (1 seul rééchantillonnage) "
                         "au lieu de masquer en 299 puis redimensionner")
parser.add_argument('--batch-size', type=int, default=None,
                    help="Override du batch size par défaut du modèle")

args = parser.parse_args()

SIZE = args.size
INPUT_SHAPE = (SIZE, SIZE, 1)

# --- Fichiers dépendants de la résolution / du run ---
IMG_ARRAY_FILE = f'img_array_{SIZE}.npy'
CROPPED_ARRAY_FILE = f'cropped_img_array_{SIZE}.npy'
CLASS_ARRAY_FILE = f'class_array_{SIZE}.npy'

# nommage par résolution et aussi si lancer via l'option o (netoyage des outlier)
SUFFIX = f"{SIZE}_o" if args.o else f"{SIZE}"
XTRAIN_FILE = f'xtrain_{SUFFIX}.pickle'
YTRAIN_FILE = f'ytrain_{SUFFIX}.pickle'
XTEST_FILE = f'xtest_{SUFFIX}.pickle'
YTEST_FILE = f'ytest_{SUFFIX}.pickle'


##################################################
# Helpers
##################################################

def resize_np_array(arr, new_size, method=None, chunk=512):
    """
    Redimensionne un array (N, H, W) ou (N, H, W, C) vers (N, new_size, new_size, ...).
    Traitement par blocs pour limiter la RAM. dtype conservé.
    method : 'nearest' pour les masques, sinon 'lanczos3' (downscale, cohérent avec
    la convention de covid19_dataprep) / 'bilinear' (upscale).
    """
    squeeze_back = (arr.ndim == 3)
    if squeeze_back:
        arr = arr[..., np.newaxis]

    if arr.shape[1] == new_size and arr.shape[2] == new_size:
        return arr[..., 0] if squeeze_back else arr

    if method is None:
        method = 'lanczos3' if arr.shape[1] > new_size else 'bilinear'

    out = np.empty((arr.shape[0], new_size, new_size, arr.shape[-1]), dtype=arr.dtype)
    for i in range(0, arr.shape[0], chunk):
        block = tf.image.resize(
            tf.convert_to_tensor(arr[i:i + chunk], dtype=tf.float32),
            (new_size, new_size),
            method=method
        ).numpy()
        if np.issubdtype(arr.dtype, np.integer):
            block = np.clip(np.round(block), 0, 255)
        out[i:i + chunk] = block.astype(arr.dtype)

    return out[..., 0] if squeeze_back else out


def augmenter_classe_batch(X_class, y_train, y_train_flat, label_value,
                           target_size, data_augmentation, batch=256):
    """
    Équivalent local de prep.augmenter_classe_numpy, mais l'augmentation se fait
    par blocs (1 appel TF par bloc au lieu d'un par image) et le dtype d'entrée
    est préservé. Défini ici pour ne pas modifier covid19_dataprep.py, partagé.
    """
    n = len(X_class)
    n_to_generate = target_size - n

    if n_to_generate <= 0:
        return X_class, y_train[y_train_flat == label_value]

    X_aug = np.empty((n_to_generate,) + X_class.shape[1:], dtype=X_class.dtype)

    done = 0
    while done < n_to_generate:
        b = min(batch, n_to_generate - done)
        idx = np.random.randint(0, n, size=b)

        block = tf.convert_to_tensor(X_class[idx], dtype=tf.float32)
        out = data_augmentation(block, training=True).numpy()

        if np.issubdtype(X_class.dtype, np.integer):
            out = np.clip(np.round(out), 0, 255)

        X_aug[done:done + b] = out.astype(X_class.dtype)
        done += b

    y_aug = np.full(len(X_aug), label_value)

    X_final = np.concatenate([X_class, X_aug], axis=0)
    y_final = np.concatenate([y_train[y_train_flat == label_value], y_aug], axis=0)

    return X_final, y_final


def load_class_arrays(class_dir, verbose_name):
    """
    Charge images + masques d'une classe et renvoie les versions dédupliquées.

    Deux stratégies :
      - purist : chaque PNG est lu et rééchantillonné une seule fois vers SIZE
                 (images: interpolation par défaut ; masques: nearest obligatoire).
      - défaut : tout est ramené à 299 (masques 256 -> 299 en nearest), le masquage
                 se fait à la résolution native des images, le redimensionnement
                 vers SIZE intervient après le crop (cf. section concaténation).
    """
    start = time.time()
    print(f'Début chargement {verbose_name}...')

    load_size = SIZE if args.purist else NATIVE_IMG_SIZE

    # images natives 299 : lanczos pour le downscale (299 -> 299 = identité, pas d'interpolation)
    img_dir = os.path.join(class_dir, 'images/')
    img_arr = prep.load_img_as_np_arr(img_dir, new_size=(load_size, load_size),
                                      interp_method='lanczos')

    # masques natifs 256 : nearest impératif (sinon bords non-binaires après interpolation)
    msk_dir = os.path.join(class_dir, 'masks/')
    msk_arr = prep.load_img_as_np_arr(msk_dir, new_size=(load_size, load_size),
                                      interp_method='nearest')

    unique_img_arr, unique_msk_arr = prep.unique_np_arr(img_arr, msk_arr)

    print('Fin :', round(time.time() - start, 3), 's',
          f'| shape img {unique_img_arr.shape} | shape msk {unique_msk_arr.shape}')
    return unique_img_arr, unique_msk_arr


##################################################
# Registre des modèles
##################################################

def save_pickle(model, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def _bs(default):
    return args.batch_size if args.batch_size is not None else default


MODEL_REGISTRY = {
    'cnn': {
        'ext': '.pickle',
        'train': lambda X, y: cnn.train_cnn_model(
            X_train=X, y_train=y,
            input_shape=INPUT_SHAPE, target_size=INPUT_SHAPE,
            epochs=500, batch_size=_bs(64), patience=50
        ),
        'save': save_pickle,
        'load': load_pickle,
        'evaluate': cnn.evaluate_cnn_model,
    },
    'densenet121': {
        'ext': '.keras',
        'train': lambda X, y: densenet.train_densenet121_model(
            X_train=X, y_train=y,
            input_shape=INPUT_SHAPE, target_size=INPUT_SHAPE,
            epochs=30, batch_size=_bs(32), patience=50,
            fine_tune_epochs=500,
            fine_tune_at=-50,            # nombre de couches dégelées en fin de base (les 50 dernières)
            head_lr=1e-3,               # learning rate phase 1 (tête, base gelée)
            finetune_lr=1e-4,           # learning rate phase 2 (fine-tuning)
            reduce_lr_patience=3,    # patience du ReduceLROnPlateau (défaut = max(patience//5, 3))
            divergence_factor=4.0,      # StopOnDivergence : seuil relatif sur val_loss
            divergence_patience=3,      # StopOnDivergence : epochs consécutifs avant arrêt
            divergence_abs_cap=1e2,     # StopOnDivergence : plafond absolu de la loss (explosion)
            monitor_metric=args.metric
        ),
        'save': lambda model, filename: model.save(filename),
        'load': tf.keras.models.load_model,
        'evaluate': lambda model, X_test, y_test: densenet.evaluate_densenet121_model(
            model=model, X_test=X_test, y_test=y_test,
            csv_path=f"predictions_densenet121_{args.n}.csv"
        ),
    },
    'efficientnetb2': {
        'ext': '.keras',
        # train_efficientnet_model renvoie (model, base_model) -> on ne garde que le modèle
        'train': lambda X, y: efficientnet.train_efficientnet_model(
            X_train=X, y_train=y,
            input_shape=INPUT_SHAPE,
            epochs=500, batch_size=_bs(32), patience=50
        )[0],
        'save': lambda model, filename: model.save(filename),
        'load': tf.keras.models.load_model,
        'evaluate': lambda model, X_test, y_test: efficientnet.evaluate_efficientnet_model(
            model=model, X_test=X_test, y_test=y_test
        ),
    },
}

model_cfg = MODEL_REGISTRY[args.model]
model_filename = f"{args.model}_{args.n}{model_cfg['ext']}"

workspace = 'data/COVID-19_Radiography_Dataset/'

print(f"[config] modèle={args.model} | size={SIZE} | purist={args.purist} | run={args.n}")

if args.i:
    ##################################################
    ##################################################
    #
    # Data preparation
    #
    ##################################################
    ##################################################

    unique_covid_img_arr, unique_covid_msk_arr = load_class_arrays(
        os.path.join(workspace, 'COVID'), 'COVID')

    unique_lung_opa_img_arr, unique_lung_opa_msk_arr = load_class_arrays(
        os.path.join(workspace, 'Lung_Opacity'), 'Lung Opacity')

    unique_normal_img_arr, unique_normal_msk_arr = load_class_arrays(
        os.path.join(workspace, 'Normal'), 'Normal')

    unique_pneumo_img_arr, unique_pneumo_msk_arr = load_class_arrays(
        os.path.join(workspace, 'Viral Pneumonia'), 'Viral Pneumonia')

    if args.o:
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

        image_level_stats_df = prep.get_image_level_pixel_stats_with_ratios(
            class_image_arrays, dark_threshold=30, bright_threshold=240)

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
        print('Fin :', round(end - start, 3), 's')

    ##################################################
    # Class arrays
    ##################################################

    # 0 = COVID
    # 1 = Lung Opacity
    # 2 = Normal
    # 3 = Pneumonia

    covid_class_array = np.full(unique_covid_img_arr.shape[0], 0)
    lung_opa_class_array = np.full(unique_lung_opa_img_arr.shape[0], 1)
    normal_class_array = np.full(unique_normal_img_arr.shape[0], 2)
    pneumo_class_array = np.full(unique_pneumo_img_arr.shape[0], 3)

    ##################################################
    # Array concatenation + masking
    ##################################################

    start = time.time()
    print('Début concaténation arrays...')

    # features
    img_array = np.concatenate([unique_covid_img_arr,
                                unique_lung_opa_img_arr,
                                unique_normal_img_arr,
                                unique_pneumo_img_arr],
                               axis=0).astype(np.uint8)

    msk_array = np.concatenate([unique_covid_msk_arr,
                                unique_lung_opa_msk_arr,
                                unique_normal_msk_arr,
                                unique_pneumo_msk_arr],
                               axis=0).astype(np.uint8)

    # binarisation stricte du masque (le nearest garantit 0/255, on sécurise quand même)
    msk_array = (msk_array > 127).astype(np.float32)

    # masquage à la résolution de chargement
    cropped_img_array = (img_array * msk_array).astype(np.uint8)
    del msk_array

    # mode par défaut : chargement/masquage en 299 -> descente vers SIZE après le crop (lanczos3)
    if not args.purist and SIZE != NATIVE_IMG_SIZE:
        start_rs = time.time()
        print(f'Redimensionnement {NATIVE_IMG_SIZE} -> {SIZE}...')
        img_array = resize_np_array(img_array, SIZE)
        cropped_img_array = resize_np_array(cropped_img_array, SIZE)
        print('Fin :', round(time.time() - start_rs, 3), 's')

    # target
    class_array = np.concatenate([covid_class_array,
                                  lung_opa_class_array,
                                  normal_class_array,
                                  pneumo_class_array],
                                 axis=0).astype(np.uint8)

    end = time.time()
    print('Fin :', round(end - start, 3), 's')
    print(f'img_array {img_array.shape} | cropped {cropped_img_array.shape} | y {class_array.shape}')

    ##################################################
    # Save preprocessed arrays
    ##################################################

    start = time.time()
    print('Début écriture fichiers...')

    np.save(IMG_ARRAY_FILE, img_array.astype(np.uint8))
    np.save(CROPPED_ARRAY_FILE, cropped_img_array.astype(np.uint8))
    np.save(CLASS_ARRAY_FILE, class_array.astype(np.uint8))

    end = time.time()
    print('Fin :', round(end - start, 3), 's')

else:

    if args.d:

        ##################################################
        # Data augmentation
        ##################################################

        start = time.time()
        print('Début chargement arrays locales...')

        img_array = np.load(IMG_ARRAY_FILE)
        cropped_img_array = np.load(CROPPED_ARRAY_FILE)
        class_array = np.load(CLASS_ARRAY_FILE)

        end = time.time()
        print('Fin :', round(end - start, 3), 's')

        # garde-fou : les .npy doivent correspondre à --size
        if cropped_img_array.shape[1] != SIZE:
            raise ValueError(
                f"{CROPPED_ARRAY_FILE} est en {cropped_img_array.shape[1]}² "
                f"alors que --size vaut {SIZE}. Relancer -i avec --size {SIZE}."
            )

        indices = np.arange(len(class_array))
        X_train, X_test, y_train, y_test = train_test_split(
            # img_array,# test 1-5
            cropped_img_array,
            class_array,
            indices,
            test_size=0.2,
            random_state=66,
            stratify=class_array
        )[:4]

        start = time.time()
        print("Génération des données augmentées en cours...")

        data_augmentation = tf.keras.Sequential([
            # layers.RandomFlip("horizontal_and_vertical"),
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

        AUG_BATCH = 512 if SIZE <= 128 else (256 if SIZE <= 256 else 128)

        X_normal_res, y_normal_res = X_normal, y_train[mask_normal]  # Inchangée
        X_covid_res, y_covid_res = augmenter_classe_batch(X_covid, y_train, y_train_flat, 0, target_size, data_augmentation, batch=AUG_BATCH)
        X_lung_opa_res, y_lung_opa_res = augmenter_classe_batch(X_lung_opa, y_train, y_train_flat, 1, target_size, data_augmentation, batch=AUG_BATCH)
        X_pneumo_res, y_pneumo_res = augmenter_classe_batch(X_pneumo, y_train, y_train_flat, 3, target_size, data_augmentation, batch=AUG_BATCH)

        X_train_balanced = np.concatenate([X_normal_res, X_covid_res, X_lung_opa_res, X_pneumo_res], axis=0)
        y_train_balanced = np.concatenate([y_normal_res, y_covid_res, y_lung_opa_res, y_pneumo_res], axis=0)

        # Reshuffle of indexes
        indexes = np.arange(len(X_train_balanced))
        np.random.shuffle(indexes)

        X_train = X_train_balanced[indexes]
        y_train = y_train_balanced[indexes]

        for obj, path in [(X_test, XTEST_FILE), (y_test, YTEST_FILE),
                          (X_train, XTRAIN_FILE), (y_train, YTRAIN_FILE)]:
            with open(path, 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
        print("Fin :", round(end - start, 3), "s")

    if not args.d:
        with open(XTEST_FILE, 'rb') as handle:
            X_test = pickle.load(handle)
        with open(YTEST_FILE, 'rb') as handle:
            y_test = pickle.load(handle)
        with open(XTRAIN_FILE, 'rb') as handle:
            X_train = pickle.load(handle)
        with open(YTRAIN_FILE, 'rb') as handle:
            y_train = pickle.load(handle)

    if args.t:
        start = time.time()
        print(f"Entraînement de {args.model} ({args.n}) en {SIZE}²...")

        model = model_cfg['train'](X_train, y_train)
        model_cfg['save'](model, model_filename)
        print(f"Modèle sauvegardé : {model_filename}")

            ##################################################
            # EfficientNetB2
            ##################################################

    if not args.t:
        start = time.time()
        print(f"Chargement du modèle {model_filename}...")

        model = model_cfg['load'](model_filename)

        with open(XTEST_FILE, 'rb') as handle:
            X_test = pickle.load(handle)
        with open(YTEST_FILE, 'rb') as handle:
            y_test = pickle.load(handle)

            end = time.time()
            print("Fin :", round(end - start, 3), "s")

        if not args.t :

            start = time.time()
            print("Chargement du modèle...")

            with open('eff_model.pickle', 'rb') as handle:
                model = pickle.load(handle)
            with open('eff_bmodel.pickle', 'rb') as handle:
                base_model = pickle.load(handle)
            with open('xtest.pickle', 'rb') as handle:
                X_test = pickle.load(handle)
            with open('ytest.pickle', 'rb') as handle:
                y_test = pickle.load(handle)

            end = time.time()
            print("Fin :", round(end - start, 3), "s")

    model_cfg['evaluate'](model=model, X_test=X_test, y_test=y_test)
