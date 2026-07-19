#!/usr/bin/env python3
"""
Harnais d'expériences pour l'optimisation de DenseNet121.

Ce script implémente la feuille de route d'optimisation SANS toucher à
covid19_main.py. Il charge les tableaux .npy (distribution NATURELLE),
refait le split reproductible (seed 66), redimensionne à la volée à la
taille voulue, applique la stratégie de déséquilibre choisie, entraîne
via covid19_densenet_utils.train_densenet121_model (version paramétrable),
évalue et sauvegarde.

Points clés alignés sur la feuille de route :
  - taille d'image testable (128 / 224 / 256) par resize à la volée
  - class_weight (balanced / boost COVID) OU oversampling, jamais les deux
  - augmentation à la volée, appliquée AU TRAIN uniquement
  - config "proxy" rapide via --quick (128², sous-échantillon, epochs réduits)

Pré-requis : img_array.npy / cropped_img_array.npy / class_array.npy
(générés par covid19_main.py -i). Par défaut on utilise le cropped.

Exemples
--------
# Round 1 — déséquilibre, config proxy rapide (128², 40% du train)
python covid19_main_densenet.py --quick --imbalance balanced   -n r1_balanced
python covid19_main_densenet.py --quick --imbalance covid_boost --covid-boost 1.5 -n r1_boost15
python covid19_main_densenet.py --quick --imbalance oversample -n r1_oversample

# Round 2 — learning rate (sur le gagnant du round 1)
python covid19_main_densenet.py --quick --imbalance balanced --head-lr 3e-4 --finetune-lr 3e-5 -n r2_lr_a

# Round 3 — taille d'image (config plus complète)
python covid19_main_densenet.py --img-size 224 --imbalance balanced -n r3_224
python covid19_main_densenet.py --img-size 256 --imbalance balanced -n r3_256

# Run final pleine fidélité
python covid19_main_densenet.py --img-size 224 --imbalance covid_boost --covid-boost 1.5 \
       --batch-size 64 --epochs 500 --patience 15 --finetune-epochs 50 --metric covid_f1 -n final
"""

import argparse
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import covid19_dataprep as prep
import covid19_densenet_utils as densenet


# ------------------------------------------------------------------ #
# Chargement + split reproductible (identique à covid19_main.py)
# ------------------------------------------------------------------ #
def load_natural_split(use_cropped=True, test_size=0.2, random_state=66):
    """
    Charge les tableaux .npy et refait EXACTEMENT le split de main.py
    (mêmes seed / stratify), donc X_test correspond au jeu de test habituel.
    Renvoie la distribution NATURELLE (non suréchantillonnée).
    """
    array_file = "cropped_img_array.npy" if use_cropped else "img_array.npy"
    print(f"Chargement de {array_file} + class_array.npy ...")
    X = np.load(array_file)
    y = np.load("class_array.npy")

    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, idx_train, idx_test


def subsample_stratified(X, y, fraction, random_state=66):
    """Sous-échantillon stratifié (pour la config proxy). fraction dans ]0,1]."""
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=fraction, random_state=random_state, stratify=y
    )
    print(f"Sous-échantillon proxy : {len(y_sub)} images ({fraction:.0%} du train)")
    return X_sub, y_sub


# ------------------------------------------------------------------ #
# Oversampling (pour la baseline, réutilise la logique de main.py)
# ------------------------------------------------------------------ #
def oversample_to_majority(X_train, y_train, augmentation):
    """
    Suréchantillonne chaque classe minoritaire jusqu'à la taille de la
    classe majoritaire (Normal), en générant des images augmentées.
    Reproduit le comportement offline de covid19_main.py, mais ici à la
    taille d'image courante (après resize).
    """
    y_flat = np.asarray(y_train).flatten()
    counts = {c: int(np.sum(y_flat == c)) for c in np.unique(y_flat)}
    target = max(counts.values())          # taille de la classe majoritaire
    print(f"Oversampling vers {target} images/classe. Comptes initiaux : {counts}")

    X_parts, y_parts = [], []
    for c in np.unique(y_flat):
        X_c = X_train[y_flat == c]
        if len(X_c) == target:
            X_parts.append(X_c)
            y_parts.append(y_train[y_flat == c])
        else:
            X_res, y_res = prep.augmenter_classe_numpy(
                X_c, y_train, y_flat, int(c), target, augmentation
            )
            X_parts.append(X_res)
            y_parts.append(y_res)

    X_bal = np.concatenate(X_parts, axis=0)
    y_bal = np.concatenate(y_parts, axis=0)

    order = np.arange(len(X_bal))
    np.random.shuffle(order)
    return X_bal[order], y_bal[order]


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def build_parser():
    p = argparse.ArgumentParser(description="Recherche d'hyperparamètres DenseNet121")

    p.add_argument("-n", default="exp", help="Nom du run (suffixe des fichiers de sortie)")
    p.add_argument("--raw", action="store_true",
                   help="Utiliser img_array.npy (non masqué) au lieu du cropped")

    # Taille / batch / lr
    p.add_argument("--img-size", type=int, default=256, choices=[128, 224, 256, 299])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-5)

    # Déséquilibre (axe unique, options exclusives)
    p.add_argument("--imbalance", default="balanced",
                   choices=["none", "balanced", "covid_boost", "oversample"],
                   help="Stratégie de gestion du déséquilibre")
    p.add_argument("--covid-boost", type=float, default=1.5,
                   help="Multiplicateur du poids COVID (mode covid_boost)")

    # Augmentation à la volée (train only) — ignorée en mode oversample
    p.add_argument("--aug", default="default",
                   choices=["none", "light", "default", "strong"])
    p.add_argument("--flip", action="store_true",
                   help="Autoriser le flip horizontal (prudence : latéralité CXR)")

    # Entraînement
    p.add_argument("--metric", default="covid_f1", choices=["macro_f1", "covid_f1"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--finetune-epochs", type=int, default=50)
    p.add_argument("--fine-tune-at", type=int, default=-50)
    p.add_argument("--no-fine-tune", action="store_true")
    p.add_argument("--head-bn", action="store_true",
                   help="Ajouter une BatchNormalization dans la tête")

    # Garde-fous anti-divergence (callback StopOnDivergence)
    p.add_argument("--divergence-factor", type=float, default=4.0,
                   help="Arrêt si val_loss dépasse factor × son minimum "
                        "pendant divergence-patience epochs (0 pour désactiver)")
    p.add_argument("--divergence-patience", type=int, default=3,
                   help="Nb d'epochs consécutifs de dérive de val_loss avant arrêt")
    p.add_argument("--divergence-abs-cap", type=float, default=1e4,
                   help="Arrêt immédiat si la loss d'entraînement dépasse ce "
                        "plafond (détecte les explosions type 20M que "
                        "TerminateOnNaN ne voit pas)")
    p.add_argument("--checkpoint-path", default=None,
                   help="Chemin du checkpoint best (défaut : best_<nom_run>.keras)")

    # Config proxy
    p.add_argument("--subsample", type=float, default=1.0,
                   help="Fraction stratifiée du train (config proxy)")
    p.add_argument("--quick", action="store_true",
                   help="Preset proxy : 128², subsample 0.4, epochs 60/patience 8, "
                        "finetune 15 (surchargé par les flags explicites)")

    return p


def apply_quick_preset(args, explicit):
    """Le preset --quick ne surcharge que ce qui n'a PAS été passé explicitement."""
    if "--img-size" not in explicit:
        args.img_size = 128
    if "--subsample" not in explicit:
        args.subsample = 0.4
    if "--epochs" not in explicit:
        args.epochs = 60
    if "--patience" not in explicit:
        args.patience = 8
    if "--finetune-epochs" not in explicit:
        args.finetune_epochs = 15
    return args


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    import sys
    parser = build_parser()
    args = parser.parse_args()

    explicit = set(a for a in sys.argv[1:] if a.startswith("--"))
    if args.quick:
        args = apply_quick_preset(args, explicit)

    size = args.img_size
    input_shape = (size, size, 1)

    print("=" * 60)
    print(f"RUN : {args.n}")
    print(f"  taille={size}  batch={args.batch_size}  "
          f"head_lr={args.head_lr}  finetune_lr={args.finetune_lr}")
    print(f"  imbalance={args.imbalance}  aug={args.aug}  flip={args.flip}  "
          f"metric={args.metric}")
    print(f"  epochs={args.epochs}  patience={args.patience}  "
          f"finetune_epochs={args.finetune_epochs}  fine_tune_at={args.fine_tune_at}")
    print(f"  divergence : factor={args.divergence_factor}  "
          f"patience={args.divergence_patience}  abs_cap={args.divergence_abs_cap:g}")
    print("=" * 60)

    t0 = time.time()

    # 1) Données naturelles + split reproductible
    X_train, X_test, y_train, y_test, idx_train, idx_test = load_natural_split(
        use_cropped=not args.raw
    )

    # 2) Sous-échantillon proxy (train uniquement)
    X_train, y_train = subsample_stratified(X_train, y_train, args.subsample)

    # 3) Resize à la volée (train + test) à la taille voulue
    print(f"Resize -> {size}x{size} ...")
    X_train = densenet.resize_image_array(X_train, size)
    X_test = densenet.resize_image_array(X_test, size)

    # 4) Stratégie de déséquilibre
    augment = None
    class_weight = None

    if args.imbalance == "oversample":
        # aug baked offline, loss non pondérée (baseline actuelle)
        aug_pipeline = densenet.build_augmentation(
            "default" if args.aug == "none" else args.aug,
            horizontal_flip=args.flip,
        )
        print("Oversampling en cours (baseline)...")
        X_train, y_train = oversample_to_majority(X_train, y_train, aug_pipeline)
        # pas de class_weight, pas d'augment à la volée
    else:
        # distribution naturelle + augmentation à la volée (train only)
        if args.aug != "none":
            augment = densenet.build_augmentation(args.aug, horizontal_flip=args.flip)
        if args.imbalance == "balanced":
            class_weight = densenet.compute_class_weights(y_train, mode="balanced")
        elif args.imbalance == "covid_boost":
            class_weight = densenet.compute_class_weights(
                y_train, mode="balanced", covid_boost=args.covid_boost
            )
        # imbalance == "none" -> class_weight reste None
        if class_weight is not None:
            print(f"class_weight = { {k: round(v, 3) for k, v in class_weight.items()} }")

    # 5) Entraînement
    model = densenet.train_densenet121_model(
        X_train=X_train,
        y_train=y_train,
        input_shape=input_shape,
        target_size=input_shape,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        monitor_metric=args.metric,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.finetune_epochs,
        fine_tune_at=args.fine_tune_at,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
        class_weight=class_weight,
        augment=augment,
        head_bn=args.head_bn,
        run_name=args.n,
        checkpoint_path=args.checkpoint_path,
        divergence_factor=args.divergence_factor,
        divergence_patience=args.divergence_patience,
        divergence_abs_cap=args.divergence_abs_cap,
    )

    # 6) Sauvegarde modèle
    model_file = f"densenet121_{args.n}.keras"
    model.save(model_file)
    print(f"Modèle sauvegardé : {model_file}")

    # 7) Sauvegarde du jeu de test à CETTE taille (pour le Grad-CAM)
    #    -> nommé par taille pour ne pas écraser les autres résolutions
    xtest_file = f"xtest_{size}.pickle"
    ytest_file = f"ytest_{size}.pickle"
    with open(xtest_file, "wb") as h:
        pickle.dump(X_test, h, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ytest_file, "wb") as h:
        pickle.dump(y_test, h, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Jeu de test sauvegardé : {xtest_file} / {ytest_file}")

    # 8) Évaluation (sans Grad-CAM interactif ; il est rejoué à part)
    densenet.evaluate_densenet121_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        show_gradcam=False,
        csv_path=f"predictions_{args.n}.csv",
        image_ids=idx_test,
    )

    print(f"\nDurée totale : {round(time.time() - t0, 1)} s "
          f"({round((time.time() - t0) / 60, 1)} min)")


if __name__ == "__main__":
    main()