import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import cv2
import pandas as pd


CLASS_NAMES = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
COVID_CLASS_INDEX = 0
BASE_MODEL_NAME = "densenet121"


##################################################
# Préparation des entrées
##################################################

def _prepare_inputs(X):
    """
    Met les images au format attendu par le modèle :
    - conversion en float32
    - ajout de la dimension canal si absente (N, 256, 256) -> (N, 256, 256, 1)
    Les valeurs restent dans [0, 255] : preprocess_input de DenseNet
    est appliqué directement dans le graphe du modèle.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
    return X


def _grayscale_to_preprocessed_rgb(inputs):
    """
    Graphe de préparation partagé entre le modèle principal et le
    modèle Grad-CAM : niveaux de gris -> 3 canaux, puis normalisation
    ImageNet propre à DenseNet (attend des valeurs 0-255 en entrée).
    """
    x = layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    x = preprocess_input(x)
    return x


##################################################
# Helpers d'optimisation (déséquilibre / augmentation / taille)
##################################################

def compute_class_weights(y, mode="balanced", covid_boost=1.0,
                          covid_index=COVID_CLASS_INDEX, normalize=True):
    """
    Calcule un dict {classe: poids} à passer à model.fit(class_weight=...).

    ATTENTION : à utiliser sur la distribution NATURELLE (dédupliquée),
    PAS sur un jeu déjà suréchantillonné. Combiner oversampling + class_weight
    sur-corrige le déséquilibre et dégrade en général.

    mode :
      - "balanced" : w_c = N / (n_classes * count_c)   (façon sklearn)
      - "none"     : tous les poids à 1
    covid_boost : multiplie le poids de la classe COVID (ex. 1.5 pour
                  pousser le recall COVID). 1.0 = aucun boost.
    normalize   : renormalise pour que la moyenne pondérée par la fréquence
                  vaille 1, afin de garder l'échelle de la loss (et donc le
                  learning rate effectif) comparable entre configurations.
    """
    y = np.asarray(y).flatten()
    classes = np.unique(y)
    n = len(y)
    k = len(classes)

    weights = {}
    for c in classes:
        count = int(np.sum(y == c))
        if mode == "balanced":
            weights[int(c)] = n / (k * count)
        elif mode == "none":
            weights[int(c)] = 1.0
        else:
            raise ValueError("mode doit être 'balanced' ou 'none'")

    if covid_index in weights:
        weights[covid_index] *= covid_boost

    if normalize:
        weighted_mean = sum(weights[int(c)] * np.sum(y == c) for c in classes) / n
        weights = {c: w / weighted_mean for c, w in weights.items()}

    return weights


def build_augmentation(strength="default", horizontal_flip=False):
    """
    Pipeline d'augmentation à appliquer UNIQUEMENT au train (à la volée).

    Choix adaptés aux radios thoraciques :
      - PAS de flip vertical (jamais anatomiquement valide)
      - flip horizontal désactivé par défaut : il casse la latéralité
        (silhouette cardiaque à gauche, marqueurs R/L). À activer avec
        prudence seulement, via horizontal_flip=True.
      - rotations volontairement PETITES : une radio est cadrée, ±10-18°
        au maximum. (Ton pipeline offline utilisait factor=0.15 ≈ ±54°,
        ce qui est très agressif pour du CXR.)

    Rappel : RandomRotation(factor) exprime l'angle en fraction de 2π,
    donc factor=0.05 ≈ ±18°.
    """
    aug = []
    if horizontal_flip:
        aug.append(layers.RandomFlip("horizontal"))

    if strength == "light":
        aug += [layers.RandomRotation(0.03),                 # ≈ ±11°
                layers.RandomZoom(0.05, 0.05),
                layers.RandomContrast(0.05)]
    elif strength == "default":
        aug += [layers.RandomRotation(0.05),                 # ≈ ±18°
                layers.RandomZoom(0.10, 0.10),
                layers.RandomContrast(0.10)]
    elif strength == "strong":
        aug += [layers.RandomRotation(0.08),                 # ≈ ±29°
                layers.RandomZoom(0.15, 0.15),
                layers.RandomContrast(0.15),
                layers.RandomTranslation(0.08, 0.08)]
    else:
        raise ValueError("strength doit être 'light', 'default' ou 'strong'")

    return tf.keras.Sequential(aug, name=f"augmentation_{strength}")


def resize_image_array(arr, size, method="lanczos3", antialias=True, chunk=2000):
    """
    Redimensionne un tableau (N, H, W) ou (N, H, W, 1) vers (N, size, size, 1),
    valeurs conservées dans [0, 255] (preprocess_input est fait dans le modèle).

    Downscale : lanczos3 + antialias, cohérent avec la préparation d'origine.
    Traitement par blocs (chunk) pour limiter la mémoire sur les gros tableaux.
    Retourne un float32.
    """
    arr = _prepare_inputs(arr)
    if arr.shape[1] == size and arr.shape[2] == size:
        return arr

    out = []
    for i in range(0, len(arr), chunk):
        r = tf.image.resize(arr[i:i + chunk], (size, size),
                            method=method, antialias=antialias)
        r = tf.clip_by_value(r, 0.0, 255.0)   # lanczos peut dépasser [0,255]
        out.append(r.numpy().astype(np.float32))

    return np.concatenate(out, axis=0)


##################################################
# Callback F1 (covid ou macro)
##################################################

class F1ScoreCallback(tf.keras.callbacks.Callback):
    """
    Calcule le f1-score sur le jeu de validation à la fin de chaque epoch
    et l'injecte dans les logs sous la clé 'val_f1', ce qui permet à
    EarlyStopping et ReduceLROnPlateau de monitorer cette métrique.

    metric :
    - 'covid_f1' : f1 de la classe COVID uniquement
    - 'macro_f1' : f1 macro (moyenne non pondérée des 4 classes)

    IMPORTANT : ce callback doit être placé AVANT EarlyStopping et
    ReduceLROnPlateau dans la liste des callbacks, sinon 'val_f1'
    n'existe pas encore dans les logs quand ils s'exécutent.
    """

    def __init__(self, X_val, y_val, metric="macro_f1", batch_size=32):
        super().__init__()
        assert metric in ("covid_f1", "macro_f1"), \
            "metric doit être 'covid_f1' ou 'macro_f1'"
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        y_proba = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        if self.metric == "covid_f1":
            score = f1_score(
                self.y_val, y_pred,
                labels=[COVID_CLASS_INDEX],
                average=None,
                zero_division=0
            )[0]
        else:
            score = f1_score(self.y_val, y_pred, average="macro", zero_division=0)

        logs["val_f1"] = score
        print(f" - val_f1 ({self.metric}) : {score:.4f}")


##################################################
# Callback d'arrêt sur divergence
##################################################

class StopOnDivergence(tf.keras.callbacks.Callback):
    """
    Coupe l'entraînement dès que la loss diverge, sans attendre la
    patience de l'EarlyStopping. Deux détections complémentaires :

    1) Explosion franche (vérifiée sur la loss d'entraînement) :
       - NaN ou inf
       - dépassement d'un plafond absolu 'abs_cap'
       Arrêt IMMÉDIAT (dès l'epoch concernée). C'est ce qui aurait tué
       en une epoch le run à loss=20 000 000, que TerminateOnNaN NE
       détecte PAS (20M est fini, ce n'est pas un NaN).

    2) Dérive progressive (vérifiée sur 'monitor', par défaut val_loss) :
       si la valeur suivie dépasse factor × sa meilleure valeur pendant
       'patience' epochs consécutives, on arrête. Filtre les pics
       transitoires (un seul mauvais epoch qui se rattrape) tout en
       coupant les vraies divergences bien avant la patience longue de
       l'EarlyStopping.

    Ce callback n'interfère pas avec restore_best_weights : quand il
    déclenche stop_training, les autres callbacks de l'epoch s'exécutent
    aussi (d'où l'intérêt de le placer APRÈS ModelCheckpoint, pour que le
    meilleur modèle soit déjà écrit sur disque à ce moment-là).
    """

    def __init__(self, monitor="val_loss", factor=4.0, patience=3, abs_cap=1e4):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.abs_cap = abs_cap
        self.best = np.inf
        self.wait = 0

    def on_train_begin(self, logs=None):
        # réinitialise l'état entre deux phases (phase 1 / fine-tuning)
        self.best = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # --- 1) explosion franche sur la loss d'entraînement ---
        train_loss = logs.get("loss")
        if train_loss is not None:
            if not np.isfinite(train_loss) or (
                self.abs_cap is not None and train_loss > self.abs_cap
            ):
                print(
                    f"\n[StopOnDivergence] loss={train_loss:.3g} "
                    f"(non finie ou > {self.abs_cap:g}) -> arrêt immédiat."
                )
                self.model.stop_training = True
                return

        # --- 2) dérive progressive sur la métrique suivie ---
        current = logs.get(self.monitor)
        if current is None or not np.isfinite(current):
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        elif self.factor > 0 and self.best > 0 and current > self.factor * self.best:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"\n[StopOnDivergence] {self.monitor}={current:.3g} "
                    f"> {self.factor}× meilleur ({self.best:.3g}) pendant "
                    f"{self.patience} epochs -> arrêt."
                )
                self.model.stop_training = True
        else:
            self.wait = 0


##################################################
# Construction du modèle
##################################################

def build_densenet121_model(input_shape=(256, 256, 1), n_classes=4, head_bn=False):
    """
    Construit un DenseNet121 en transfer learning :
    - entrée en niveaux de gris (1 canal), dupliquée sur 3 canaux
    - preprocess_input DenseNet intégré au modèle
    - base ImageNet gelée par défaut
    - tête de classification : GAP -> Dropout -> Dense -> softmax
    Retourne (model, base_model) pour pouvoir dégeler la base ensuite.
    """
    inputs = layers.Input(shape=input_shape)

    x = _grayscale_to_preprocessed_rgb(inputs)

    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3)
    )
    base_model._name = BASE_MODEL_NAME
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    if head_bn:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="densenet121_covid")
    return model, base_model


##################################################
# Entraînement
##################################################

def train_densenet121_model(
    X_train,
    y_train,
    input_shape=(256, 256, 1),
    target_size=(256, 256, 1),   # conservé pour garder la même signature que le CNN
    epochs=500,
    batch_size=64,
    patience=50,
    validation_split=0.2,
    monitor_metric="macro_f1",   # 'macro_f1' ou 'covid_f1'
    fine_tune=True,
    fine_tune_epochs=50,
    fine_tune_at=-50,            # nombre de couches dégelées en fin de base (les 50 dernières)
    head_lr=1e-3,               # learning rate phase 1 (tête, base gelée)
    finetune_lr=1e-5,           # learning rate phase 2 (fine-tuning)
    class_weight=None,          # dict {classe: poids} ; à utiliser SUR distribution naturelle
    augment=None,               # callable d'augmentation appliqué AU TRAIN uniquement (à la volée)
    head_bn=False,              # ajoute une BatchNormalization dans la tête
    reduce_lr_patience=None,    # patience du ReduceLROnPlateau (défaut = max(patience//5, 3))
    random_state=66,
    run_name=None,              # nom du run -> nom du fichier checkpoint best_{run_name}.keras
    checkpoint_path=None,       # chemin explicite du checkpoint (prioritaire sur run_name)
    divergence_factor=4.0,      # StopOnDivergence : seuil relatif sur val_loss
    divergence_patience=3,      # StopOnDivergence : epochs consécutifs avant arrêt
    divergence_abs_cap=1e4,     # StopOnDivergence : plafond absolu de la loss (explosion)
):
    """
    Entraîne un DenseNet121 en deux phases :
    1) base ImageNet gelée, seule la tête de classification apprend
    2) (optionnel, fine_tune=True) dégel des dernières couches de la base
       avec un learning rate réduit

    L'early stopping et la réduction du learning rate sont pilotés par
    'val_f1', calculé par le callback F1ScoreCallback selon monitor_metric :
    - 'covid_f1' : maximise le f1 de la classe COVID
    - 'macro_f1' : maximise le f1 macro

    Robustesse (nouveau) :
    - ModelCheckpoint : le meilleur modèle (val_f1 max) est écrit en
      continu sur disque (best_{run_name}.keras). Le même fichier est
      partagé entre la phase 1 et le fine-tuning : la phase 2 n'écrase le
      fichier que si elle bat le meilleur val_f1 de la phase 1
      (via initial_value_threshold). À la fin, le modèle retourné est
      rechargé depuis ce fichier -> c'est le meilleur des deux phases,
      quelle que soit la façon dont le run s'est terminé.
    - StopOnDivergence : coupe immédiatement si la loss explose
      (NaN/inf ou > divergence_abs_cap), et coupe en quelques epochs si
      val_loss dérive (> divergence_factor × son minimum pendant
      divergence_patience epochs). Évite d'attendre la patience complète
      de l'EarlyStopping sur un run condamné.
    """
    X_train = _prepare_inputs(X_train)
    y_train = np.asarray(y_train).flatten()

    # Split de validation manuel et stratifié : nécessaire pour que le
    # callback F1 dispose explicitement des données de validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_train
    )

    model, base_model = build_densenet121_model(
        input_shape=input_shape,
        n_classes=len(np.unique(y_train)),
        head_bn=head_bn,
    )

    if reduce_lr_patience is None:
        reduce_lr_patience = max(patience // 5, 3)

    # Chemin du checkpoint : explicite > dérivé du run_name > défaut générique
    if checkpoint_path is None:
        if run_name is not None:
            checkpoint_path = f"best_{run_name}.keras"
        else:
            checkpoint_path = "best_densenet121.keras"

    # Entrée d'entraînement : si 'augment' est fourni, on construit un pipeline
    # tf.data qui applique l'augmentation AU TRAIN UNIQUEMENT (le val reste en
    # numpy, non augmenté). Sinon, chemin numpy identique à l'origine.
    if augment is not None:
        AUTOTUNE = tf.data.AUTOTUNE
        shuffle_buffer = min(len(X_tr), 10000)
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            .shuffle(shuffle_buffer, seed=random_state, reshuffle_each_iteration=True)
            .batch(batch_size)
            .map(lambda a, b: (augment(a, training=True), b),
                 num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE)
        )
    else:
        train_ds = None

    def _fit(n_epochs, initial_threshold=None):
        # class_weight fonctionne aussi bien avec un tf.data (x, y) qu'avec numpy
        if train_ds is not None:
            return model.fit(
                train_ds,
                validation_data=(X_val, y_val),
                epochs=n_epochs,
                callbacks=make_callbacks(initial_threshold),
                class_weight=class_weight,
                verbose=1,
            )
        return model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=n_epochs,
            batch_size=batch_size,
            callbacks=make_callbacks(initial_threshold),
            class_weight=class_weight,
            verbose=1,
        )

    def make_callbacks(initial_threshold=None):
        # Ordre important :
        #   1. F1ScoreCallback   -> remplit logs['val_f1']
        #   2. ModelCheckpoint   -> lit val_f1, écrit le meilleur sur disque
        #   3. StopOnDivergence  -> après le checkpoint : si on coupe sur
        #      divergence, le meilleur modèle est déjà sauvegardé
        #   4. EarlyStopping / ReduceLROnPlateau -> lisent val_f1
        return [
            F1ScoreCallback(X_val, y_val, metric=monitor_metric, batch_size=batch_size),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_f1",
                mode="max",
                save_best_only=True,
                # ne pas écraser en phase 2 tant qu'on n'a pas battu la phase 1
                initial_value_threshold=initial_threshold,
                verbose=1,
            ),
            StopOnDivergence(
                monitor="val_loss",
                factor=divergence_factor,
                patience=divergence_patience,
                abs_cap=divergence_abs_cap,
            ),
            EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_f1",
                mode="max",
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            ),
        ]

    def _best_val_f1(history):
        # meilleur val_f1 observé sur une phase (pour seuiller la phase 2)
        vals = (history.history or {}).get("val_f1", [])
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return max(vals) if vals else None

    ##################################################
    # Phase 1 : base gelée
    ##################################################
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    print(f"Phase 1 : entraînement de la tête (base gelée), métrique suivie : {monitor_metric}...")

    hist1 = _fit(epochs)
    best_f1 = _best_val_f1(hist1)

    ##################################################
    # Phase 2 : fine-tuning des dernières couches
    ##################################################
    if fine_tune:
        print("Phase 2 : fine-tuning des dernières couches de la base...")

        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        # Les couches BatchNormalization restent gelées (stabilité)
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # On seuille le checkpoint avec le meilleur val_f1 de la phase 1 :
        # le fine-tuning ne remplace le fichier que s'il fait strictement
        # mieux. Ainsi, même si la phase 2 diverge d'entrée, le fichier
        # conserve le meilleur modèle de la phase 1.
        hist2 = _fit(fine_tune_epochs, initial_threshold=best_f1)
        best_f1 = _best_val_f1(hist2) if _best_val_f1(hist2) is not None else best_f1

    ##################################################
    # Rechargement du meilleur modèle depuis le disque
    ##################################################
    # ModelCheckpoint a écrit le meilleur modèle (toutes phases confondues)
    # dans checkpoint_path. On le recharge : c'est l'assurance contre un
    # arrêt sur divergence, où restore_best_weights n'est pas garanti.
    import os
    if os.path.exists(checkpoint_path):
        try:
            best_model = tf.keras.models.load_model(checkpoint_path, safe_mode=False)
            print(f"Meilleur modèle rechargé depuis : {checkpoint_path} "
                  f"(val_f1 ≈ {best_f1:.4f})" if best_f1 is not None
                  else f"Meilleur modèle rechargé depuis : {checkpoint_path}")
            return best_model
        except Exception as e:
            print(f"[warn] Rechargement du checkpoint impossible ({e}). "
                  f"On garde les poids en mémoire (restore_best_weights).")

    return model


##################################################
# Interpretability : Grad-CAM
##################################################
# Adapté de la version CNN, mais DenseNet121 étant un sous-modèle
# imbriqué, on ne peut pas faire model.get_layer('conv...') directement.
# On découpe le modèle en deux :
#   - feature_model : image -> dernière carte de features de la base
#     (sortie de DenseNet121 include_top=False = après le ReLU final,
#     c'est la couche cible standard pour Grad-CAM sur DenseNet)
#   - classifier_model : carte de features -> prédictions (la tête,
#     avec les poids déjà entraînés)
# Le gradient est ensuite calculé à la jonction des deux.

def _make_gradcam_models(model, base_model_name=BASE_MODEL_NAME):
    base_model = model.get_layer(base_model_name)

    # 1) feature_model : on rejoue la préparation partagée puis la base
    inputs = layers.Input(shape=model.input_shape[1:])
    x = _grayscale_to_preprocessed_rgb(inputs)
    features = base_model(x, training=False)
    feature_model = models.Model(inputs, features)

    # 2) classifier_model : on rejoue les couches de tête (mêmes objets,
    # donc mêmes poids) sur une nouvelle entrée de la taille des features
    head_input = layers.Input(shape=feature_model.output_shape[1:])
    y = head_input
    passed_base = False
    for layer in model.layers:
        if layer.name == base_model_name:
            passed_base = True
            continue
        if passed_base:
            y = layer(y)
    classifier_model = models.Model(head_input, y)

    return feature_model, classifier_model


def get_gradcam_heatmap(model, image, class_idx, base_model_name=BASE_MODEL_NAME):
    """
    Calcule la heatmap Grad-CAM pour une image (H, W) ou (H, W, 1),
    par rapport à la classe class_idx.
    Même logique que la version CNN : gradients de la sortie de classe
    par rapport à la dernière carte de features, pondération des canaux
    par la moyenne des gradients, ReLU, normalisation [0, 1].
    """
    feature_model, classifier_model = _make_gradcam_models(model, base_model_name)

    img = _prepare_inputs(np.expand_dims(np.squeeze(image), axis=0))
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = feature_model(img_tensor, training=False)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradient None : vérifiez le nom du sous-modèle de base.")

    grads = grads[0]
    local_conv_outputs = conv_outputs[0]

    # Poids = moyenne globale des gradients par canal
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Combinaison linéaire des canaux, puis ReLU
    cam = tf.reduce_sum(tf.multiply(weights, local_conv_outputs), axis=-1)
    cam = tf.nn.relu(cam).numpy()

    # Normalisation [0, 1]
    if cam.max() > 0:
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    return cam


def show_gradcam_overlay(input_image, heatmap, true_class=None, pred_class=None,
                         alpha=0.5, image_id=None, proba=None, target_class=None,
                         save_path=None):
    """
    Affiche l'image originale, la heatmap Grad-CAM et la superposition.
    Gère les images en [0, 255] (uint8) comme en [0, 1].

    image_id     : identifiant affiché dans le titre (position dans X_test
                   ou id d'origine issu du CSV)
    proba        : vecteur des 4 probabilités softmax. Si fourni, un 4e
                   panneau affiche la distribution complète en barres.
                   La confidence (max) est aussi ajoutée au titre.
    target_class : classe sur laquelle le gradient a été pris, si elle
                   diffère de la classe prédite (mode contrefactuel).
    save_path    : si fourni, enregistre la figure au lieu de seulement
                   l'afficher (utile pour les figures du rapport).
    """
    input_image = np.asarray(input_image, dtype=np.float32)
    if input_image.ndim == 2:
        input_image = np.expand_dims(input_image, axis=-1)

    # Normalisation en [0, 1] pour l'affichage si nécessaire
    if input_image.max() > 1.0:
        input_image = input_image / 255.0

    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_u8, (input_image.shape[1], input_image.shape[0]))
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    if input_image.shape[-1] == 1:
        img_uint8 = np.uint8(255 * input_image.squeeze())
        input_image_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        input_display = input_image.squeeze()
        cmap = 'gray'
    elif input_image.shape[-1] == 3:
        input_image_rgb = np.uint8(255 * input_image)
        input_display = input_image
        cmap = None
    else:
        raise ValueError("L'image doit avoir 1 ou 3 canaux.")

    overlay_bgr = cv2.addWeighted(heatmap_colored_bgr, alpha, input_image_rgb, 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # ----- Titre -----
    t = CLASS_NAMES[true_class] if isinstance(true_class, (int, np.integer)) else true_class
    p = CLASS_NAMES[pred_class] if isinstance(pred_class, (int, np.integer)) else pred_class

    header = "Grad-CAM (DenseNet121)"
    if image_id is not None:
        header += f" — image #{image_id}"

    line2 = ""
    if true_class is not None or pred_class is not None:
        line2 = f"True: {t}  |  Predicted: {p}"
        if proba is not None and pred_class is not None:
            line2 += f" (confidence {np.asarray(proba)[pred_class]:.3f})"
        # Marqueur visuel : la bonne ou la mauvaise réponse
        if true_class is not None and pred_class is not None:
            line2 += "  ✓" if int(true_class) == int(pred_class) else "  ✗"

    line3 = ""
    if target_class is not None and pred_class is not None and int(target_class) != int(pred_class):
        tc = CLASS_NAMES[target_class] if isinstance(target_class, (int, np.integer)) else target_class
        line3 = f"[contrefactuel] gradient calculé sur : {tc}"

    title_text = "\n".join([s for s in (header, line2, line3) if s])

    # ----- Figure -----
    # GridSpec plutôt que subplot : les panneaux d'images ont un aspect
    # verrouillé (imshow), donc tight_layout ne détecte pas le débordement
    # des étiquettes du 4e panneau. On supprime les étiquettes d'axe et on
    # écrit le nom de la classe DANS la barre : plus rien ne peut déborder.
    n_panels = 4 if proba is not None else 3
    width_ratios = [1, 1, 1, 0.85] if proba is not None else [1, 1, 1]

    fig = plt.figure(figsize=(5 * n_panels, 5))
    gs = fig.add_gridspec(1, n_panels, width_ratios=width_ratios, wspace=0.15)
    fig.suptitle(title_text, fontsize=13)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Original")
    ax1.imshow(input_display, cmap=cmap)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Grad-CAM Heatmap")
    ax2.imshow(heatmap_colored)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Grad-CAM Overlay")
    ax3.imshow(overlay)
    ax3.axis('off')

    if proba is not None:
        proba = np.asarray(proba).flatten()
        ax4 = fig.add_subplot(gs[0, 3])

        # Vert = vraie classe, rouge = classe prédite si elle est fausse,
        # gris = les autres. Lecture immédiate de la nature de l'erreur.
        colors = []
        for i in range(len(proba)):
            if true_class is not None and i == int(true_class):
                colors.append("tab:green")
            elif pred_class is not None and i == int(pred_class):
                colors.append("tab:red")
            else:
                colors.append("lightgray")

        y_pos = np.arange(len(proba))
        ax4.barh(y_pos, proba, color=colors, height=0.6)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([])          # aucune étiquette à gauche de l'axe
        ax4.invert_yaxis()
        ax4.set_xlim(0, 1)
        ax4.set_xlabel("Probabilité")
        ax4.set_title("Distribution softmax")
        ax4.spines[['top', 'right']].set_visible(False)

        # Nom de la classe + valeur, écrits dans la barre si elle est assez
        # longue, sinon juste à sa droite (cas des probabilités quasi nulles)
        for i, v in enumerate(proba):
            label = f"{CLASS_NAMES[i]}  {v:.3f}"
            if v > 0.45:
                ax4.text(v - 0.02, i, label, va="center", ha="right",
                         fontsize=9, color="white", fontweight="bold")
            else:
                ax4.text(v + 0.02, i, label, va="center", ha="left",
                         fontsize=9, color="0.25")

    fig.subplots_adjust(top=0.86)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure enregistrée : {save_path}")

    plt.show()


##################################################
# Export des prédictions
##################################################

def export_predictions_csv(y_true, y_pred, y_proba, image_ids=None,
                           class_names=CLASS_NAMES, filename="predictions.csv"):
    """
    Écrit un CSV avec, pour chaque image du jeu de test :
    - image_id : identifiant de l'image (voir note ci-dessous)
    - y_true / y_true_name : classe réelle (indice + nom)
    - y_pred / y_pred_name : classe prédite (indice + nom)
    - correct : booléen (prédiction juste ou non)
    - confidence : probabilité de la classe prédite (max softmax)
    - proba_<classe> : une colonne de probabilité par classe

    image_ids :
    - si None, on utilise la position dans X_test (0..N-1). C'est
      suffisant pour recroiser avec le Grad-CAM, qui indexe le même
      X_test.
    - si fourni (ex. idx_test tracé à travers le train_test_split),
      l'id renvoie à la position dans le tableau concaténé d'origine,
      donc reproductible d'un run à l'autre.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_proba = np.asarray(y_proba)

    if image_ids is None:
        image_ids = np.arange(len(y_true))
    image_ids = np.asarray(image_ids).flatten()

    data = {
        "image_id": image_ids,
        "y_true": y_true,
        "y_true_name": [class_names[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_name": [class_names[i] for i in y_pred],
        "correct": (y_true == y_pred),
        "confidence": y_proba.max(axis=1),
    }
    for j, name in enumerate(class_names):
        col = "proba_" + name.replace(" ", "_")
        data[col] = y_proba[:, j]

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Prédictions exportées : {filename} ({len(df)} lignes)")
    return df


##################################################
# Évaluation
##################################################

def evaluate_densenet121_model(model, X_test, y_test, batch_size=32,
                               show_gradcam=True, gradcam_img_idx=10,
                               csv_path=None, image_ids=None):
    """
    Évalue le modèle sur le jeu de test :
    - accuracy / loss
    - classification report (précision, rappel, f1 par classe)
    - matrice de confusion
    - f1 macro et f1 COVID
    - Grad-CAM sur une image du jeu de test (comme la version CNN)
    """
    X_test = _prepare_inputs(X_test)
    y_test = np.asarray(y_test).flatten()

    loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print(f"Loss test     : {loss:.4f}")
    print(f"Accuracy test : {acc:.4f}")

    y_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    print("\nClassification report :")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    f1_covid = f1_score(y_test, y_pred, labels=[COVID_CLASS_INDEX], average=None, zero_division=0)[0]
    print(f"\nF1 macro : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 COVID : {f1_covid:.4f}")

    if csv_path is not None:
        export_predictions_csv(y_test, y_pred, y_proba,
                               image_ids=image_ids, filename=csv_path)

    if show_gradcam:
        input_image = X_test[gradcam_img_idx]
        true_label = int(y_test[gradcam_img_idx])
        pred_class = int(y_pred[gradcam_img_idx])
        img_id = image_ids[gradcam_img_idx] if image_ids is not None else gradcam_img_idx

        heatmap = get_gradcam_heatmap(model, input_image, pred_class)
        show_gradcam_overlay(input_image, heatmap,
                             true_class=true_label, pred_class=pred_class,
                             image_id=img_id, proba=y_proba[gradcam_img_idx])

    return y_pred