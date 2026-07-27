import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2


#################################
# Interpretability
#################################

# https://medium.com/@oveis/easy-guide-using-gradcam-algorithm-to-explain-cnn-classification-of-sar-images-mstar-database-470973bb1174


def get_gradcam_heatmap(model, image, class_idx, layer_name):
    # 1. Récupération du sous modèle

    base_net = model.get_layer('efficientnetb2')
    
    # 2. Récupération de la dernière couche de convolution
    conv_layer = base_net.get_layer(layer_name)
    
    # 3. Modèle intermédiaire qui part de l'entrée du sous-modèle (3 canaux) ; renvoie les activations de 'top_conv' ET la sortie brute d'EfficientNet
    base_grad_model = models.Model(
        inputs=[base_net.input],
        outputs=[conv_layer.output, base_net.output]
    )
    
    # Préparation de l'image (1 canal)
    img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0) # shape: (1, 256, 256, 1)

    # Enregistrement des opérations
    with tf.GradientTape() as tape:
        # Étape A : On passe l'image 1 canal dans votre couche ConvRGB (1x1) du modèle global
        conv_rgb_layer = model.get_layer('ConvRGB')
        img_rgb = conv_rgb_layer(img_tensor) # shape: (1, 256, 256, 3)
        
        # Étape B : On passe l'image RGB dans notre modèle de gradient interne
        conv_outputs, base_outputs = base_grad_model(img_rgb)
        
        # Étape C : On applique manuellement la fin du modèle global (Dropout + Dense)
        dropout_layer = model.get_layer('Dropout')
        outputs_layer = model.get_layer('Outputs')
        
        final_outputs = dropout_layer(base_outputs, training=False)
        predictions = outputs_layer(final_outputs)
        
        # Perte pour la classe d'intérêt
        loss = predictions[:, class_idx]

    # 4. Calcul du gradient de la perte par rapport aux activations de 'top_conv'
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError(f"Le gradient est None. Vérifiez les connexions des couches.")

    # Passage du format batch (1, H, W, C) au format 3D (H, W, C)
    grads = grads[0]
    local_conv_outputs = conv_outputs[0]

    # Calcul des poids (moyenne globale des gradients par canal de la carte de caractéristiques)
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # Combinaison linéaire des canaux de la carte de caractéristiques
    cam = tf.reduce_sum(tf.multiply(weights, local_conv_outputs), axis=-1)

    # Passage par la fonction ReLU (on ne garde que les caractéristiques qui augmentent la décision)
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


def create_efficientnet_model(
    input_shape = (256,256,1)
):

    # 1. Entrée du modèle
    inputs = layers.Input(shape=input_shape, name = 'Inputs')

    ### !!! PAS DE RESCALING POUR EfficientNet avec Keras !!!

    # 2. Conv 1x1 qui apprend à mapper le grayscale vers le RGB
    x = layers.Conv2D(3, (1, 1), padding='same', name = 'ConvRGB')(inputs)

    # 3. Modèle de base EfficientNet-B2 (qui reçoit maintenant 3 canaux)
    base_model = tf.keras.applications.EfficientNetB2(
        include_top=False, 
        weights='imagenet', 
        input_shape=(256, 256, 3), # Reçoit la sortie de la Conv2D
        pooling='avg'
    )
    x = base_model(x)

    # 4. Couche de dropout
    x = layers.Dropout(0.3, name = 'Dropout')(x) 

    # 5. Couche de sortie
    outputs = layers.Dense(4, activation='softmax', name = 'Outputs')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name = 'EfficientNetB2')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    return model, base_model


def train_efficientnet_model(
    X_train,
    y_train,
    # y, 
    input_shape,
    epochs,
    batch_size,
    patience
):
    
    efficientnet_model, base_model = create_efficientnet_model(input_shape)

    base_model.trainable = False

    early_stopping = EarlyStopping(
        monitor='val_loss',       # On surveille la perte sur les données de validation
        patience=patience,        # Si pendant X epochs consécutives la perte ne baisse plus, on arrête
        restore_best_weights=True # récupère les meilleurs poids, pas ceux de la dernière epoch
    )

    # weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y), y = y)

    training_history_efficientnet_frozen = efficientnet_model.fit(X_train
                                         , y_train
                                         , validation_split = 0.2
                                         , epochs = 5 # faible pour fixer la couche Dense
                                         , batch_size = batch_size
                                         , callbacks = [early_stopping]
                                         , shuffle = True
                                        # , class_weight = weights
                                        )
    base_model.trainable = True

    efficientnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    training_history_efficientnet_unfrozen = efficientnet_model.fit(X_train
                                         , y_train
                                         , validation_split = 0.2
                                         , epochs = epochs # faible pour fixer la couche Dense
                                         , batch_size = batch_size
                                         , callbacks = [early_stopping]
                                         , shuffle = True
                                        # , class_weight = weights
                                        )

    return efficientnet_model, base_model


def evaluate_efficientnet_model(
    model,
    X_test,
    y_test,
    img_idx
):
    test_pred_efficientnet = model.predict(X_test)
    test_pred_efficientnet_class = np.argmax(test_pred_efficientnet, axis=1)

    class_report = metrics.classification_report(y_test, test_pred_efficientnet_class, output_dict= True)

    input_image = X_test[img_idx]  # shape: (256, 256, 1)
    true_label = y_test[img_idx]
    
    preds = model.predict(np.expand_dims(input_image, axis=0))
    pred_class = np.argmax(preds[0])

    target_layer_name = 'top_conv'

    heatmap = get_gradcam_heatmap(model, input_image, pred_class, layer_name=target_layer_name)

    fig = show_gradcam_overlay(input_image, heatmap, true_class=true_label, pred_class=pred_class)

    return fig, class_report