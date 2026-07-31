import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

CLASS_NAMES = ["COVID", "Lung Opacity", "Normal", "Viral Pneumonia"]
COVID_CLASS_INDEX = 0
#BASE_MODEL_NAME = "efficientnetb2"

def create_cnn_model(
    input_shape=(256, 256, 1)
    , target_size=(128, 128)
):
    """
    Crée un CNN avec une couche de resize dynamique.
    
    input_shape : Taille des images en entrée
    target_size : Taille des images à tester
    """

    # 1. Entrée du modèle
    inputs = layers.Input(shape=input_shape, name='Inputs')
    
    # 2. Normalisation (Rescaling)
    x = layers.Rescaling(1.0 / 255, name='Rescaling')(inputs)

    # --- BLOCS DE RÉDUCTION DYNAMIQUE ---
    filters = 32
    current_dim = input_shape[0]
    target_dim = target_size[0]
    
    i = 1
    while current_dim > target_dim:
        x = layers.Conv2D(filters, (3, 3), strides=(2, 2), activation='relu', padding='same', name=f'Resize_{i}')(x)
        current_dim = current_dim // 2
        filters = min(filters * 2, 256)
        i += 1
    
    # 4. Bloc Convolutif 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='Conv1')(x)
    x = layers.MaxPooling2D((2, 2), name='MaxPool1')(x)
    
    # 5. Bloc Convolutif 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='MaxPool2')(x)
    
    # 6. Bloc Convolutif 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv3')(x)
    x = layers.MaxPooling2D((2, 2), name='MaxPool3')(x)

    # 7. Classification (Couches denses)
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dense(128, activation='relu', name='Dense1')(x)
    x = layers.Dropout(0.5, name='Dropout')(x)
    
    # 8. Couche de sortie
    outputs = layers.Dense(4, activation='softmax', name='Outputs')(x)
    
    # Création finale du modèle fonctionnel
    model = models.Model(inputs=inputs, outputs=outputs, name='Functional_CNN')
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model



#################################
# Interpretability
#################################

# https://medium.com/@oveis/easy-guide-using-gradcam-algorithm-to-explain-cnn-classification-of-sar-images-mstar-database-470973bb1174


def get_gradcam_heatmap(model, image, class_idx, layer_name):
    # Récupération de la couche convolutive cible
    conv_layer = model.get_layer(layer_name)
    
    grad_model = models.Model(
        inputs=[model.input],
        outputs=[conv_layer.output, model.output]
    )

    # Préparation de l'image (Tenseur en Float32)
    img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0) 

    # Enregistrement des opérations pour le calcul du gradient
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    # Extraction du gradient de la perte par rapport aux sorties de Conv3
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError(f"Le gradient est toujours None. Vérifiez que '{layer_name}' est bien le nom de la couche.")

    # On passe du format batch (1, H, W, C) au format 3D (H, W, C)
    grads = grads[0]
    local_conv_outputs = conv_outputs[0]

    # Calcul des poids (moyenne globale par canal)
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Combinaison linéaire des canaux de la carte de caractéristiques
    cam = tf.reduce_sum(tf.multiply(weights, local_conv_outputs), axis=-1)

    # Passage par la fonction ReLU
    cam = tf.nn.relu(cam).numpy()

    # Normalisation finale [0, 1]
    if cam.max() > 0:
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
    else:
        cam = np.zeros_like(cam)

    return cam


def show_gradcam_overlay(input_image, heatmap, true_class=None, pred_class=None, alpha=0.5):
    """
    Displays original input, Grad-CAM heatmap, and overlay with class info.
    
    Args:
        input_image (np.array): Input image (H, W, 1) or (H, W, 3)
        heatmap (np.array): Grad-CAM heatmap (range 0–1)
        true_class (str or int): True class label (optional)
        pred_class (str or int): Predicted class label (optional)
        alpha (float): Blending factor for overlay
    """
    # Scale and resize heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Prepare input image and convert to 3-channel RGB
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
        raise ValueError("Input image must have 1 or 3 channels.")

    # Create overlay image
    overlay_bgr = cv2.addWeighted(heatmap_colored_bgr, alpha, input_image_rgb, 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Build figure title
    title_text = "Grad-CAM Visualization"
    if true_class is not None or pred_class is not None:
        title_text += f"\nTrue: {true_class} | Predicted: {pred_class}"

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle(title_text, fontsize=14)

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(input_display, cmap=cmap)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap_colored)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Grad-CAM Overlay")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    return fig


#################################
#################################
#################################


def train_cnn_model(
    X_train,
    y_train,
    # y, 
    input_shape,
    target_size,
    epochs,
    batch_size,
    patience
):
    
    cnn_model = create_cnn_model(input_shape, target_size)

    early_stopping = EarlyStopping(
        monitor='val_loss',       # On surveille la perte sur les données de validation
        patience=patience,        # Si pendant X epochs consécutives la perte ne baisse plus, on arrête
        restore_best_weights=True # récupère les meilleurs poids, pas ceux de la dernière epoch
    )

    # weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)

    training_history_cnn = cnn_model.fit(X_train
                                         , y_train
                                         , validation_split = 0.2
                                         , epochs = epochs
                                         , batch_size = batch_size
                                         , callbacks = [early_stopping]
                                        # , class_weight = weights
                                        )

    return cnn_model




def evaluate_cnn_model(
    model,
    X_test,
    y_test,
    img_idx
):
    test_pred_cnn = model.predict(X_test)
    test_pred_cnn_class = np.argmax(test_pred_cnn, axis = 1)

    class_report = metrics.classification_report(y_test, test_pred_cnn_class, output_dict=True)
    # affichage classification_report lors de l'appel evaluate_efficientnet_model sur covid19_main.py
    print("\nClassification report :")
    print(metrics.classification_report(y_test, test_pred_cnn_class, output_dict=False))
    #print(pd.DataFrame(class_report).T)
    # affichage Matrice de confusion lors de l'appel evaluate_efficientnet_model sur covid19_main.py
    print("Matrice de confusion :")
    print(metrics.confusion_matrix(y_test, test_pred_cnn_class))

    #img_idx = 10
    input_image = X_test[img_idx]  # shape: (H, W, 1)
    true_label = y_test[img_idx]
    preds = model.predict(np.expand_dims(input_image, axis=0))
    pred_class = np.argmax(preds[0])

    last_conv_layer = "Conv3"

    # Get heatmap
    heatmap = get_gradcam_heatmap(model, input_image, pred_class, last_conv_layer)

    fig = show_gradcam_overlay(input_image, heatmap, true_class=true_label, pred_class=pred_class)

    return fig, class_report

def show_gradcam_overlay_softmax(input_image, heatmap, true_class=None, pred_class=None,
                                 alpha=0.5, image_id=None, proba=None, target_class=None,
                                 save_path=None):
    """
    Displays original input, Grad-CAM heatmap, and overlay with class info.
    
    Args:
        input_image (np.array): Input image (H, W, 1) or (H, W, 3)
        heatmap (np.array): Grad-CAM heatmap (range 0–1)
        true_class (str or int): True class label (optional)
        pred_class (str or int): Predicted class label (optional)
        alpha (float): Blending factor for overlay
    """
    # Scale and resize heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Prepare input image and convert to 3-channel RGB
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
        raise ValueError("Input image must have 1 or 3 channels.")

    # Create overlay image
    overlay_bgr = cv2.addWeighted(heatmap_colored_bgr, alpha, input_image_rgb, 1 - alpha, 0)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # ----- Titre -----
    t = CLASS_NAMES[true_class] if isinstance(true_class, (int, np.integer)) else true_class
    p = CLASS_NAMES[pred_class] if isinstance(pred_class, (int, np.integer)) else pred_class

    # Build figure title
    #title_text = "Grad-CAM (EfficientNetB2)"
    header = "Grad-CAM (CNN)"
    # if true_class is not None or pred_class is not None:
    #     title_text += f"\nTrue: {true_class} | Predicted: {pred_class}"

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

    #ajout pour corrigé
    # nombre de lignes réelles du titre (header + éventuel line2 + éventuel line3)
    n_title_lines = title_text.count("\n") + 1
    title_h = 0.08 + 0.045 * n_title_lines   # hauteur allouée, adaptative

    #fig = plt.figure(figsize=(5 * n_panels, 5))
    fig = plt.figure(figsize=(5 * n_panels, 5 + title_h * 5))
    #gs = fig.add_gridspec(1, n_panels, width_ratios=width_ratios, wspace=0.15)
    gs = fig.add_gridspec(
        2, n_panels,
        height_ratios=[title_h, 1 - title_h],
        width_ratios=width_ratios,
        wspace=0.15, hspace=0.05
    )
     # fig.suptitle(title_text, fontsize=13)

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.set_title("Original")
    # ax1.imshow(input_display, cmap=cmap)
    # ax1.axis('off')

    # ax2 = fig.add_subplot(gs[0, 1])
    # ax2.set_title("Grad-CAM Heatmap")
    # ax2.imshow(heatmap_colored)
    # ax2.axis('off')

    # ax3 = fig.add_subplot(gs[0, 2])
    # ax3.set_title("Grad-CAM Overlay")
    # ax3.imshow(overlay)
    # ax3.axis('off')
    # --- Titre : axe dédié, indépendant des titres de subplots ---
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=13)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title("Original")
    ax1.imshow(input_display, cmap=cmap)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title("Grad-CAM Heatmap")
    ax2.imshow(heatmap_colored)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_title("Grad-CAM Overlay")
    ax3.imshow(overlay)
    ax3.axis('off')

    if proba is not None:
        proba = np.asarray(proba).flatten()
        #ax4 = fig.add_subplot(gs[0, 3])
        ax4 = fig.add_subplot(gs[1, 3])

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

    #fig.subplots_adjust(top=0.86)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure enregistrée : {save_path}")

    #plt.show() #ne marche pas pour streamlit
    return fig #ajouté pour streamlit

def explain(model,
    X_test,
    y_test,
    img_idx,
    class_idx=None
):
    """
        Affiche le Grad-CAM pour l'image à la position idx dans X_test.
        class_idx : classe cible du Grad-CAM. Si None, on prend la classe
        prédite (comportement par défaut : "pourquoi cette prédiction ?").
        image_ids : optionnel, tableau des ids d'origine (idx_test). Si absent,
        l'id affiché est la position dans X_test.
    """
    input_image = X_test[img_idx]  # shape: (256, 256, 1)
    true_label = y_test[img_idx]
    
    x = np.expand_dims(input_image, axis=0)
    proba = model.predict(x, verbose=0)[0]
    pred_class = int(np.argmax(proba))

    target = class_idx if class_idx is not None else pred_class

    target_layer_name = 'Conv3'

    heatmap = get_gradcam_heatmap(model, input_image, target, layer_name=target_layer_name)

    #fig = show_gradcam_overlay(input_image, heatmap, true_class=true_label, pred_class=pred_class)
    fig = show_gradcam_overlay_softmax(input_image, heatmap,
                                true_class=true_label, pred_class=pred_class, 
                                image_id=img_idx, proba=proba, target_class=target)

    return fig