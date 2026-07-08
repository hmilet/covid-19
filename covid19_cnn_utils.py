import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2



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
    plt.figure(figsize=(15, 5))
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
    plt.show()


#################################
#################################
#################################


def train_cnn_model(
    X_train,
    y_train,
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

    training_history_cnn = cnn_model.fit(X_train, y_train, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = [early_stopping])

    return cnn_model




def evaluate_cnn_model(
    model,
    X_test,
    y_test
):
    test_pred_cnn = model.predict(X_test)
    test_pred_cnn_class = np.argmax(test_pred_cnn, axis = 1)

    print(metrics.classification_report(y_test, test_pred_cnn_class))

    img_idx = 10
    input_image = X_test[img_idx]  # shape: (H, W, 1)
    true_label = y_test[img_idx]
    preds = model.predict(np.expand_dims(input_image, axis=0))
    pred_class = np.argmax(preds[0])

    last_conv_layer = "Conv3"

    # Get heatmap
    heatmap = get_gradcam_heatmap(model, input_image, pred_class, last_conv_layer)

    show_gradcam_overlay(input_image, heatmap, true_class=true_label, pred_class=pred_class)