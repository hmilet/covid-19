import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def flatten_if_needed(X):
    """
    Transforme les images en vecteurs si elles ne sont pas déjà en 2D.

    Exemple :
    (3616, 256, 256, 1) devient (3616, 65536)

    Si X est déjà sous forme :
    (n_samples, n_features)
    alors la fonction ne change rien.
    """

    X = np.asarray(X)

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    return X


def get_default_svm_param_grid():
    """
    Retourne une grille de paramètres par défaut pour le SVM.

    Les paramètres testés sont :
    - kernel
    - C
    - gamma
    - degree uniquement pour le kernel poly
    """

    param_grid = [
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10]
        },
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", "auto", 0.001, 0.01]
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", "auto"],
            "svm__degree": [2, 3]
        }
    ]

    return param_grid

def get_default_pca_svm_param_grid():
    """
    Grille légère pour tester PCA + SVM.
    On commence simple pour éviter de refaire planter la machine.
    """

    param_grid = [
        {
            "pca__n_components": [50, 100, 150],
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10]
        }
    ]

    return param_grid

def train_evaluate_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid=None,
    scoring="f1_macro",
    cv=5,
    scale=True,
    target_names=None,
    n_jobs=-1,
    verbose=1
):
    """
    Entraîne et évalue un modèle SVM multiclasses.

    Cette fonction :
    1. transforme les images en vecteurs si besoin ;
    2. cherche les meilleurs paramètres avec GridSearchCV ;
    3. entraîne le meilleur modèle ;
    4. évalue le modèle sur le jeu de test.

    Paramètres
    ----------
    X_train : array-like
        Données d'entraînement.
        Peut être sous forme image : (n, h, w, c)
        ou déjà vectorisée : (n, features).

    y_train : array-like
        Labels d'entraînement.

    X_test : array-like
        Données de test.

    y_test : array-like
        Labels de test.

    param_grid : list ou dict, optionnel
        Grille de paramètres pour GridSearchCV.
        Si None, une grille par défaut est utilisée.

    scoring : str
        Métrique utilisée pour sélectionner le meilleur modèle.
        Par défaut : "f1_macro".

    cv : int
        Nombre de folds pour la validation croisée.

    scale : bool
        Si True, applique StandardScaler avant le SVM.

    target_names : list, optionnel
        Noms des classes pour le classification_report.

    n_jobs : int
        Nombre de cœurs utilisés.
        -1 signifie utiliser tous les cœurs disponibles.

    verbose : int
        Niveau d'affichage de GridSearchCV.

    Retour
    ------
    results : dict
        Dictionnaire contenant :
        - best_model
        - best_params
        - best_cv_score
        - accuracy
        - macro_precision
        - macro_recall
        - macro_f1
        - per_class_metrics
        - classification_report
        - confusion_matrix
        - cv_results
        - y_pred
    """

    # -----------------------------
    # 1. Préparation des données
    # -----------------------------
    X_train = flatten_if_needed(X_train)
    X_test = flatten_if_needed(X_test)

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    # -----------------------------
    # 2. Noms des classes
    # -----------------------------
    if target_names is None:
        target_names = [
            "COVID",
            "Lung Opacity",
            "Normal",
            "Pneumonia"
        ]

    # -----------------------------
    # 3. Grille de paramètres
    # -----------------------------
    if param_grid is None:
        param_grid = get_default_svm_param_grid()

    # -----------------------------
    # 4. Pipeline scaler + SVM
    # -----------------------------
    if scale:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC())
        ])
    else:
        model = Pipeline([
            ("svm", SVC())
        ])

    # -----------------------------
    # 5. Validation croisée stratifiée
    # -----------------------------
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    # -----------------------------
    # 6. Recherche des meilleurs paramètres
    # -----------------------------
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # -----------------------------
    # 7. Meilleur modèle
    # -----------------------------
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # -----------------------------
    # 8. Prédiction sur le test set
    # -----------------------------
    y_pred = best_model.predict(X_test)

    # -----------------------------
    # 9. Métriques globales
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # -----------------------------
    # 10. Métriques par classe
    # -----------------------------
    precision_per_class = precision_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    recall_per_class = recall_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    f1_per_class = f1_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    labels = np.unique(y_test)

    per_class_metrics = pd.DataFrame({
        "class_label": labels,
        "class_name": target_names[:len(labels)],
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1_score": f1_per_class
    })

    # -----------------------------
    # 11. Classification report
    # -----------------------------
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    # -----------------------------
    # 12. Matrice de confusion
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    # -----------------------------
    # 13. Résultats de GridSearchCV
    # -----------------------------
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values(
        by="rank_test_score"
    ).reset_index(drop=True)

    # -----------------------------
    # 14. Dictionnaire final
    # -----------------------------
    results = {
        "best_model": best_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,

        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "per_class_metrics": per_class_metrics,
        "classification_report": report,
        "confusion_matrix": cm,

        "cv_results": cv_results,
        "y_pred": y_pred
    }

    return results

def train_evaluate_pca_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid=None,
    scoring="f1_macro",
    cv=3,
    target_names=None,
    n_jobs=1,
    verbose=1,
    random_state=42
):
    """
    Entraîne et évalue un modèle PCA + SVM multiclasses.

    Étapes :
    1. Flatten des images si besoin
    2. Normalisation simple en float32
    3. PCA pour réduire la dimension
    4. SVM
    5. GridSearchCV
    6. Évaluation sur le test set
    """

    # -----------------------------
    # 1. Préparation des données
    # -----------------------------
    X_train = flatten_if_needed(X_train)
    X_test = flatten_if_needed(X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Si les pixels sont entre 0 et 255, on normalise entre 0 et 1
    if X_train.max() > 1:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()

    # -----------------------------
    # 2. Noms des classes
    # -----------------------------
    if target_names is None:
        target_names = [
            "COVID",
            "Lung Opacity",
            "Normal",
            "Pneumonia"
        ]

    # -----------------------------
    # 3. Grille de paramètres
    # -----------------------------
    if param_grid is None:
        param_grid = get_default_pca_svm_param_grid()

    # -----------------------------
    # 4. Pipeline PCA + SVM
    # -----------------------------
    model = Pipeline([
        (
            "pca",
            PCA(
                svd_solver="randomized",
                whiten=True,
                random_state=random_state
            )
        ),
        (
            "svm",
            SVC()
        )
    ])

    # -----------------------------
    # 5. Validation croisée stratifiée
    # -----------------------------
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    # -----------------------------
    # 6. GridSearchCV
    # -----------------------------
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    # -----------------------------
    # 7. Meilleur modèle
    # -----------------------------
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # -----------------------------
    # 8. Prédictions
    # -----------------------------
    y_pred = best_model.predict(X_test)

    # -----------------------------
    # 9. Métriques globales
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # -----------------------------
    # 10. Métriques par classe
    # -----------------------------
    precision_per_class = precision_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    recall_per_class = recall_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    f1_per_class = f1_score(
        y_test,
        y_pred,
        average=None,
        zero_division=0
    )

    labels = np.unique(y_test)

    per_class_metrics = pd.DataFrame({
        "class_label": labels,
        "class_name": [target_names[int(label)] for label in labels],
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1_score": f1_per_class
    })

    # -----------------------------
    # 11. Classification report
    # -----------------------------
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    # -----------------------------
    # 12. Matrice de confusion
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    # -----------------------------
    # 13. Résultats CV
    # -----------------------------
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results = cv_results.sort_values(
        by="rank_test_score"
    ).reset_index(drop=True)

    # -----------------------------
    # 14. Variance expliquée par la PCA
    # -----------------------------
    best_pca = best_model.named_steps["pca"]
    explained_variance_ratio = best_pca.explained_variance_ratio_.sum()

    results = {
        "best_model": best_model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,

        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,

        "per_class_metrics": per_class_metrics,
        "classification_report": report,
        "confusion_matrix": cm,

        "explained_variance_ratio": explained_variance_ratio,

        "cv_results": cv_results,
        "y_pred": y_pred
    }

    return results