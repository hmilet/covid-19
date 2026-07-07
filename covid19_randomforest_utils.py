import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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

def get_default_randomforest_param_grid():
    """
    Retourne une grille de paramètres par défaut pour le RandomForestClassifier pour utilisation GridSearchCV.

    Les paramètres testés sont :
    - n_estimators (nombre d'arbres dans la forêt)
    - max_features (nombre de pixels à regarder à chaque split)
    - max_depth (profondeur maximal de chaque arbre)
    - min_samples_split (nombre mini d'échantillons pour créer un noeud interne)
    - min_samples_leaf (nombre mini d'échantillons pour créer une feuille)
    """

    param_grid = {
        'pca__n_components': [300],
        'rf__n_estimators': [100, 200, 300, 400, 500],
        'rf__max_features': ['sqrt', 'log2', 0.2, 0.5],
        'rf__max_depth': [None, 10, 20, 30, 40],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }


    return param_grid


def train_evaluate_pca_randomforest(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid=None,
    scoring="f1_macro",
    cv=5,
    target_names=None,
    n_jobs=-1,
    verbose=2,
    random_state=42
):
    """
    Entraîne et évalue un modèle RandomForestClassifier 

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
        param_grid = get_default_randomforest_param_grid()

    # -----------------------------
    # 4. Instanciation du modèle
    # -----------------------------   

    #model = RandomForestClassifier()
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
            "rf",
            RandomForestClassifier(
                random_state=random_state
                , class_weight='balanced'
            )
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
    # 6. Recherche des meilleurs paramètres
    # -----------------------------

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,             
        cv=cv_strategy,      
        #scoring='accuracy',
        n_jobs=-n_jobs,             
        verbose=verbose,             
        random_state=random_state 
    )

    random_search.fit(X_train, y_train)

    # -----------------------------
    # 7. Meilleur modèle
    # -----------------------------
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

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
    cv_results = pd.DataFrame(random_search.cv_results_)
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


def test_pca_randomforest(
    X_train,
    y_train,
    X_test,
    y_test,
    #scoring_mode="covid",
    cv=3,
    n_jobs=-1,
    verbose=1
):
    """
    Teste un modèle PCA + RandomForestClassifier sur les images de radiographie.

    Objectif :
    - Réduire la dimension des images avec PCA.
    - Entraîner un RandomForestClassifier.
    - Chercher les meilleurs paramètres avec GridSearchCV.
    - Évaluer le modèle sur le jeu de test.

    Paramètres
    ----------
    X_train : array-like
        Images d'entraînement.
        Peut être sous forme :
        - (n_images, hauteur, largeur, canaux)
        - ou déjà aplatie : (n_images, n_features)

    y_train : array-like
        Labels d'entraînement.

    X_test : array-like
        Images de test.

    y_test : array-like
        Labels de test.

    scoring_mode : str
        Métrique utilisée pour choisir le meilleur modèle.
        
        - "covid" : optimise le F1-score de la classe COVID uniquement.
        - "macro" : optimise le F1-score macro sur les 4 classes.

    cv : int
        Nombre de folds pour la validation croisée.

    n_jobs : int
        Nombre de cœurs utilisés par GridSearchCV.
        Pour éviter les problèmes mémoire, on garde n_jobs=1.

    verbose : int
        Niveau d'affichage de GridSearchCV.

    Retour
    ------
    pca_randomforest_results : dict
        Dictionnaire contenant :
        - best_model
        - best_params
        - best_cv_score
        - explained_variance_ratio
        - accuracy
        - macro_precision
        - macro_recall
        - macro_f1
        - classification_report
        - confusion_matrix
        - cv_results
        - y_pred
    """

    start = time.time()
    print("Début test PCA + randomforest...")

    # --------------------------------------------------
    # 1. Choix de la métrique de validation croisée
    # --------------------------------------------------
    # Ici, on peut choisir entre :
    # - F1-score COVID uniquement
    # - F1-score macro sur toutes les classes

    # if scoring_mode == "covid":
    #     # Optimisation du F1-score uniquement pour la classe COVID.
    #     # La classe COVID correspond au label 0.
    #     scoring = make_scorer(
    #         f1_score,
    #         labels=[0],
    #         average="macro",
    #         zero_division=0
    #     )

    # elif scoring_mode == "macro":
    #     # Optimisation du F1-score moyen sur les 4 classes.
    #     scoring = "f1_macro"

    # else:
    #     raise ValueError("scoring_mode doit être 'covid' ou 'macro'.")

    # --------------------------------------------------
    # 2. Grille de paramètres PCA + RandomForestClassifier
    # --------------------------------------------------
    # pca__n_components = 150 :
    # on garde 150 composantes principales.

    param_grid = {
        'pca__n_components': [300],
        'rf__n_estimators': [100, 200, 300, 400, 500],
        'rf__max_features': ['sqrt', 'log2', 0.2, 0.5],
        'rf__max_depth': [None, 10, 20, 30, 40],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # --------------------------------------------------
    # 3. Entraînement + évaluation PCA + randomforest
    # --------------------------------------------------

    pca_randomforest_results = train_evaluate_pca_randomforest(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        param_grid=param_grid,
        #scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    # --------------------------------------------------
    # 4. Affichage des meilleurs paramètres
    # --------------------------------------------------

    print("Best params :", pca_randomforest_results["best_params"])
    print("Best CV score :", pca_randomforest_results["best_cv_score"])
    print("Variance expliquée PCA :", pca_randomforest_results["explained_variance_ratio"])

    # --------------------------------------------------
    # 5. Affichage des métriques globales
    # --------------------------------------------------

    print("Accuracy :", pca_randomforest_results["accuracy"])
    print("Macro precision :", pca_randomforest_results["macro_precision"])
    print("Macro recall :", pca_randomforest_results["macro_recall"])
    print("Macro F1 :", pca_randomforest_results["macro_f1"])

    # --------------------------------------------------
    # 6. Affichage du rapport détaillé
    # --------------------------------------------------

    print(pca_randomforest_results["classification_report"])
    print(pca_randomforest_results["confusion_matrix"])

    end = time.time()
    print("Fin :", round(end - start, 3), "s")

    return pca_randomforest_results