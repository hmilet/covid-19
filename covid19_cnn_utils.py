import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics



def create_cnn_model(
    input_shape=(256, 256, 1)
    , target_size=(128, 128)
):
    """
    Crée un CNN avec une couche de resize dynamique.
    
    input_shape : Taille des images en entrée
    target_size : Taille des images à tester
    """
    
    model = models.Sequential()
    
    # 1. Couche d'entrée
    model.add(layers.Input(shape=input_shape))
    
    # 2. Couche de Resize dynamique
    # Permet de tester différentes tailles simplement en changeant 'target_size'
    # model.add(layers.Resizing(target_size[0], target_size[1], interpolation='bicubic')) -> bug GPU

    # --- BLOCS DE RÉDUCTION DYNAMIQUE ---
    # On boucle tant que la dimension actuelle est plus grande que la cible
    # Chaque passage applique une convolution avec stride=2 (qui divise la taille par 2)
    filters = 32
    
    current_dim = input_shape[0]
    target_dim = target_size[0]

    while current_dim > target_dim:
        model.add(layers.Conv2D(filters, (3, 3), strides=(2, 2), activation='relu', padding='same'))
        current_dim = current_dim // 2
        filters = min(filters * 2, 256) # Augmente les filtre
    
    # 3. Normalisation (Rescaling)
    # On ramène les pixels entre 0 et 1 pour aider la convergence
    model.add(layers.Rescaling(1.0 / 255))
    
    # 4. Bloc Convolutif 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 5. Bloc Convolutif 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 6. Bloc Convolutif 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # 7. Passage aux couches denses (Classification)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Pour éviter le surapprentissage (overfitting)
    
    # 8. Couche de sortie
    # Classification multiclasse
    model.add(layers.Dense(4, activation='softmax'))
    
    loss_function = 'sparse_categorical_crossentropy'
        
    # Compilation du modèle
    model.compile(optimizer='adam',
                  loss=loss_function,
                  metrics=['accuracy'])
    
    return model

def train_evaluate_cnn_model(
    X_train,
    y_train,
    X_test,
    y_test,
    input_shape,
    target_size,
    epochs,
    batch_size
):
    
    cnn_model = create_cnn_model(input_shape, target_size)

    early_stopping = EarlyStopping(
        monitor='val_loss',       # On surveille la perte sur les données de validation
        patience=30,              # Si pendant 10 epochs consécutives la perte ne baisse plus, on arrête
        restore_best_weights=True # Très important : récupère les meilleurs poids, pas ceux de la dernière epoch
    )

    training_history_cnn = cnn_model.fit(X_train, y_train, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = [early_stopping])

    test_pred_cnn = cnn_model.predict(X_test)
    test_pred_cnn_class = np.argmax(test_pred_cnn, axis = 1)

    print(metrics.classification_report(y_test, test_pred_cnn_class))