# Classification de radiographies thoraciques (COVID-19)

Classification d'images de radiographies pulmonaires en 4 classes — **COVID**, **opacité
pulmonaire**, **normal**, **pneumonie virale** — par transfer learning sur architectures
pré-entraînées, avec comparaison à des baselines de machine learning classique.

Projet réalisé en binôme dans le cadre de la formation Machine Learning Engineer
(Liora / Mines Paris – PSL).

---

## Résultats

| Modèle | Accuracy | F1 macro | F1 COVID | Rappel COVID |
|---|---|---|---|---|
| **EfficientNetB2** (retenu) | **0.932** | **0.930** | **0.922** | **0.921** |
| DenseNet121 | 0.895 | 0.893 | 0.827 | 0.815 |
| CNN from scratch | 0.800 | 0.780 | 0.560 | 0.491 |
| SVM (baseline) | — | 0.73 | 0.50 | 0.44 |
| Random Forest (baseline) | — | 0.65 | 0.41 | 0.31 |

Rapports de classification complets (précision, rappel, F1 pour chacune des 4 classes) :
`efficientnetb2_final_classification_report.csv`,
`densenet121_final_classification_report.csv`,
`cnn_final_classification_report.csv`.

**Métrique privilégiée : le rappel sur la classe COVID.** Dans un contexte de dépistage,
un faux négatif — un cas COVID classé comme normal — coûte plus cher qu'un faux positif,
qui sera écarté par un examen complémentaire. C'est aussi la métrique qui discrimine le
plus les modèles, alors que l'accuracy globale masque les écarts.

C'est sur ce critère qu'EfficientNetB2 se détache : il rattrape 11 points de rappel COVID
sur DenseNet121 (0.921 contre 0.815) pour 3,7 points d'accuracy globale seulement. Le CNN
entraîné de zéro atteint 80 % d'accuracy mais ne détecte qu'un cas COVID sur deux (rappel
0.491), ce qui illustre l'apport du transfer learning sur un jeu de données de cette
taille.

Les baselines de ML classique confirment la difficulté de la classe COVID : rappel de
0.31 pour Random Forest et 0.44 pour SVM, malgré une optimisation des hyperparamètres
par recherche sur grille.

---

## Approche

1. **Préparation des données** — chargement, nettoyage, détection d'outliers,
   rééquilibrage des classes, augmentation.
2. **Baselines ML classique** — Random Forest et SVM après réduction de dimension par
   PCA (500 composantes / 90 % de variance pour le premier, 150 composantes / 80 % pour
   le second), validation croisée en K-fold stratifié, recherche d'hyperparamètres par
   `RandomizedSearchCV` et `GridSearchCV`.
3. **Transfer learning** — DenseNet121 et EfficientNetB2 pré-entraînés sur ImageNet,
   entraînement en deux phases (tête de classification gelée, puis fine-tuning des
   couches profondes à learning rate réduit).
4. **Interprétabilité** — Grad-CAM pour visualiser les zones de l'image qui portent la
   décision du modèle, et vérifier qu'il s'appuie sur des régions pulmonaires
   plausibles plutôt que sur des artefacts d'acquisition.
5. **Démonstrateur** — interface Streamlit : exploration du jeu de données, comparaison
   des modèles, prédiction sur une image avec heatmap Grad-CAM.

---

## Stack

**Deep learning** — TensorFlow / Keras (DenseNet121, EfficientNetB2)
**ML classique** — scikit-learn (SVM, Random Forest)
**Data & viz** — pandas, NumPy, Matplotlib, Seaborn
**Application** — Streamlit
**Environnement** — Python, gestion des dépendances avec `uv`

Le projet a été développé sur deux configurations matérielles : macOS Apple Silicon
(`tensorflow-metal`) et Linux/Windows NVIDIA (CUDA). Le `requirements.txt` utilise des
marqueurs `sys_platform` pour gérer les deux environnements depuis un fichier unique.

---

## Installation

```bash
git clone https://github.com/hmilet/covid-19.git
cd covid-19

# avec uv (recommandé)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# ou avec pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
# Entraînement DenseNet121
python covid19_main_densenet.py

# Génération des visualisations Grad-CAM
python covid19_main_gradcam_densenet.py

# Application de démonstration
streamlit run streamlit_app.py
```

---

## Structure

```
covid19_dataprep.py              préparation et augmentation des données
covid19_dataviz.py               visualisations et exploration
covid19_densenet_utils.py        modèle DenseNet121
covid19_efficientnet_utils.py    modèle EfficientNetB2
covid19_cnn_utils.py             CNN from scratch
covid19_svm_utils.py             baseline SVM
covid19_randomforest_utils.py    baseline Random Forest
covid19_main*.py                 scripts d'entraînement
streamlit_app.py                 application de démonstration
covid-19_Datasets.ipynb          notebook d'exploration
```

---

## Données

Le jeu de données n'est pas versionné dans ce dépôt. *(À compléter : nom du dataset,
source, nombre d'images par classe, lien de téléchargement.)*

---

## Limites

Ce projet a une visée pédagogique. Les performances obtenues sur ce jeu de données ne
préjugent pas d'une utilisation clinique : la validation d'un outil d'aide au diagnostic
exige des protocoles, des jeux de données multi-centriques et une évaluation par des
professionnels de santé qui sortent du cadre de ce travail.

---

## Auteurs

Projet réalisé en binôme — [hmilet](https://github.com/hmilet),
[mbenrabah](https://github.com/mbenrabah).
