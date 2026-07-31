"""
Exploration Grad-CAM autonome.

But : visualiser le Grad-CAM sur des images du jeu de test SANS relancer
tout le pipeline main.py (pas de préparation de données, pas de ré-split).
On recharge seulement le modèle .keras et les pickles xtest/ytest.

Exemples d'utilisation :
    # une image précise (position dans X_test, toutes classes confondues)
    python covid19_main_gradcam_densenet.py --model densenet121 -n cropped_covid_f1 --idx 42
    # idem, avec image size 299
    python covid19_main_gradcam_densenet.py -n purist299 --size 299 --idx 15

    # 3 images dont la VRAIE classe est COVID  <-- c'est ça qu'on veut en général
    python covid19_main_gradcam_densenet.py -n cropped_covid_f1 --true-class 0 --n-images 3

    # idem, tirées au hasard parmi les COVID
    python covid19_main_gradcam_densenet.py -n cropped_covid_f1 --true-class 0 --random
    # avec image size 299
    python covid19_main_gradcam_densenet.py -n purist299 --size 299 --true-class 0 --random

    # ATTENTION : --class-idx ne choisit PAS l'image, il choisit la classe
    # sur laquelle on dérive le gradient. Ci-dessous : sur une image COVID,
    # montrer les zones qui poussent le modèle vers "Normal" (contrefactuel).
    python covid19_main_gradcam_densenet.py -n cropped_covid_f1 --true-class 0 --class-idx 2

    # rejouer les erreurs listées dans un predictions.csv
    python covid19_main_gradcam_densenet.py -n cropped_covid_f1 --errors --csv predictions_cropped_covid_f1.csv

    # mode interactif : demande un index à la volée
    python covid19_main_gradcam_densenet.py -n cropped_covid_f1

En notebook, importer directement :
    import covid19_main_gradcam_densenet as gx
    model = gx.load_model("densenet121", "cropped_covid_f1")
    X_test, y_test = gx.load_test_data()
    gx.explain(model, X_test, y_test, idx=42)
"""

import argparse
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd

import covid19_densenet_utils as densenet
from sklearn import metrics

CLASS_NAMES_LOCAL = densenet.CLASS_NAMES


##################################################
# Chargement
##################################################

def load_test_data(xtest_path="xtest.pickle", ytest_path="ytest.pickle"):
    with open(xtest_path, "rb") as h:
        X_test = pickle.load(h)
    with open(ytest_path, "rb") as h:
        y_test = pickle.load(h)
    return X_test, np.asarray(y_test).flatten()


def load_model(model_name="densenet121", run_name="run1"):
    filename = f"{model_name}_{run_name}.keras"
    print(f"Chargement du modèle : {filename}")
    return tf.keras.models.load_model(filename)


##################################################
# Explication d'une image
##################################################

def explain(model, X_test, y_test, idx, class_idx=None, image_ids=None, save_path=None):#, base_model_name=None):
    """
    Affiche le Grad-CAM pour l'image à la position idx dans X_test.
    class_idx : classe cible du Grad-CAM. Si None, on prend la classe
    prédite (comportement par défaut : "pourquoi cette prédiction ?").
    image_ids : optionnel, tableau des ids d'origine (idx_test). Si absent,
    l'id affiché est la position dans X_test.
    """
    image = X_test[idx]
    true_label = int(y_test[idx])

    x = densenet._prepare_inputs(np.expand_dims(np.squeeze(image), axis=0))
    proba = model.predict(x, verbose=0)[0]
    pred_class = int(np.argmax(proba))

    target = class_idx if class_idx is not None else pred_class
    img_id = image_ids[idx] if image_ids is not None else idx

    print(f"Image {img_id} | vraie classe : {densenet.CLASS_NAMES[true_label]} "
          f"| prédite : {densenet.CLASS_NAMES[pred_class]} "
          f"(conf. {proba[pred_class]:.3f}) | Grad-CAM sur : {densenet.CLASS_NAMES[target]}")

    heatmap = densenet.get_gradcam_heatmap(model, image, target)#, base_model_name)
    # ajout fig= pour le streamlit
    fig = densenet.show_gradcam_overlay(image, heatmap,
                                  true_class=true_label, pred_class=pred_class,
                                  image_id=img_id, proba=proba,
                                  target_class=target, save_path=save_path)
    # jout class_report pour le streamlit
    # y_pred = model.predict(X_test)
    # y_pred_class = np.argmax(y_pred, axis=1)
    # class_report = metrics.classification_report(y_test, y_pred_class, output_dict= True)
    return fig#, class_report #ajouté pour streamlit

def find_indices(y_test, true_class=None, n=5, random=False, seed=66):
    """
    Renvoie les positions dans X_test des images dont la VRAIE classe
    est true_class. C'est ce qui permet de choisir *quelle image* on
    regarde (à ne pas confondre avec class_idx, qui choisit seulement
    la classe cible du gradient sur une image déjà choisie).
    """
    y_test = np.asarray(y_test).flatten()
    if true_class is None:
        positions = np.arange(len(y_test))
    else:
        positions = np.where(y_test == true_class)[0]

    if len(positions) == 0:
        raise ValueError(f"Aucune image de classe {true_class} dans X_test.")

    if random:
        rng = np.random.default_rng(seed)
        positions = rng.permutation(positions)

    return positions[:n].tolist()


def explain_class(model, X_test, y_test, true_class, n=3,
                  class_idx=None, random=False):
    """
    Affiche le Grad-CAM sur n images dont la vraie classe est true_class.
    Par défaut, le gradient est pris sur la classe prédite.
    """
    positions = find_indices(y_test, true_class=true_class, n=n, random=random)
    print(f"Images de classe {CLASS_NAMES_LOCAL[true_class]} : positions {positions}")
    for pos in positions:
        fig = explain(model, X_test, y_test, pos, class_idx=class_idx)
        densenet.display_gradcam(fig)


def explain_errors(model, X_test, y_test, csv_path="predictions.csv",
                   n=5, class_idx=None):
    """
    Rejoue le Grad-CAM sur les images mal classées listées dans le CSV.
    Le CSV est écrit dans l'ordre de X_test, donc l'indice de ligne du
    DataFrame correspond directement à la position dans X_test.
    """
    df = pd.read_csv(csv_path)
    wrong_positions = df.index[~df["correct"]].tolist()
    print(f"{len(wrong_positions)} erreurs trouvées dans {csv_path}. "
          f"Affichage des {min(n, len(wrong_positions))} premières.")
    for pos in wrong_positions[:n]:
        fig = explain(model, X_test, y_test, pos, class_idx=class_idx)
        densenet.display_gradcam(fig)


##################################################
# CLI
##################################################

def main():
    parser = argparse.ArgumentParser(description="Exploration Grad-CAM autonome")
    parser.add_argument("--model", default="densenet121",
                        help="Nom du modèle (préfixe du fichier .keras)")
    parser.add_argument("-n", default="run1",
                        help="Nom du run (suffixe du fichier .keras)")
    parser.add_argument("--idx", type=int, default=None,
                        help="QUELLE image : position exacte dans X_test")
    parser.add_argument("--true-class", type=int, default=None,
                        help="QUELLE image : sélectionne des images dont la VRAIE classe "
                             "est celle-ci (0=COVID,1=Lung Opacity,2=Normal,3=Pneumonia)")
    parser.add_argument("--n-images", type=int, default=3,
                        help="Nombre d'images à afficher en mode --true-class")
    parser.add_argument("--random", action="store_true",
                        help="Tirer les images au hasard plutôt que les premières (--true-class)")
    parser.add_argument("--class-idx", type=int, default=None,
                        help="SUR QUOI on dérive : classe cible du gradient Grad-CAM. "
                             "Ne sélectionne PAS l'image. Par défaut = classe prédite.")
    parser.add_argument("--errors", action="store_true",
                        help="Rejouer les erreurs depuis un predictions.csv")
    parser.add_argument("--csv", default="predictions.csv",
                        help="Chemin du CSV de prédictions (mode --errors)")
    parser.add_argument("--n-errors", type=int, default=5,
                        help="Nombre d'erreurs à afficher en mode --errors")
    parser.add_argument("--size", type=int, default=None,
                    help="Charge xtest_{size}.pickle / ytest_{size}.pickle")
    args = parser.parse_args()

    

    model = load_model(args.model, args.n)
    if args.size is not None:
        X_test, y_test = load_test_data(f"xtest_{args.size}.pickle", f"ytest_{args.size}.pickle")
    else:
        X_test, y_test = load_test_data()

    if args.errors:
        explain_errors(model, X_test, y_test, csv_path=args.csv,
                       n=args.n_errors, class_idx=args.class_idx)
    elif args.true_class is not None:
        explain_class(model, X_test, y_test, true_class=args.true_class,
                      n=args.n_images, class_idx=args.class_idx,
                      random=args.random)
    elif args.idx is not None:
        fig = explain(model, X_test, y_test, args.idx, class_idx=args.class_idx)
        densenet.display_gradcam(fig)
    else:
        # Mode interactif
        print("Mode interactif. Entrer un index d'image, 'q' pour quitter.")
        while True:
            s = input("Index image : ").strip()
            if s.lower() in ("q", "quit", "exit"):
                break
            try:
                fig = explain(model, X_test, y_test, int(s), class_idx=args.class_idx)
                densenet.display_gradcam(fig)
            except (ValueError, IndexError) as e:
                print(f"Entrée invalide : {e}")


if __name__ == "__main__":
    main()