import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def display_img_np_arr(img_arr):
    '''
    Display the type, dimensions and minimum and maximum values of a numpy array
    '''
    print(type(img_arr))
    print(img_arr.dtype)
    print(img_arr.shape)
    print(f"Value range: [{img_arr.min()}, {img_arr.max()}]")   

def get_pixel_stats(class_image_arrays):
    '''
    This function returns stats about the pixels once stored in a numpy array
    '''
    
    stats = []

    for class_name, arr in class_image_arrays.items():
        stats.append({
            "class": class_name,
            "n_images": arr.shape[0],
            "pixel_min": arr.min(),
            "pixel_max": arr.max(),
            "pixel_mean": arr.mean(),
            "pixel_std": arr.std()
        })

    return pd.DataFrame(stats)

def get_image_level_pixel_stats(class_image_arrays):
    
    stats = []

    for class_name, arr in class_image_arrays.items():
        for i in range(arr.shape[0]):
            img = arr[i]
            
            stats.append({
                "class": class_name,
                "image_index": i,
                "pixel_mean": img.mean(),
                "pixel_std": img.std()
            })

    return pd.DataFrame(stats)

def get_ratio_thresholds_by_class(stats_suspicious_df):
    """
    Retourne les seuils de ratios utilisés pour chaque classe.
    """
    
    thresholds_df = (
        stats_suspicious_df
        .groupby("class")[["dark_ratio_threshold", "bright_ratio_threshold"]]
        .first()
        .reset_index()
    )
    
    return thresholds_df

def get_suspicious_summary_by_class(stats_suspicious_df):
    """
    Retourne le nombre d'images suspectes par classe.
    """
    
    summary_df = (
        stats_suspicious_df
        .groupby("class")[[
            "is_dark_suspicious",
            "is_bright_suspicious",
            "is_suspicious"
        ]]
        .sum()
        .astype(int)
        .reset_index()
    )
    
    return summary_df

def display_image_stats(stats_df, class_image_arrays, n=12, title="Suspicious images"):
    """
    Affiche les n premières images d'un DataFrame de statistiques.
    class_image_arrays: RGB ou grey sont bien géré 
    """
    df_to_display = stats_df.head(n).copy()
    
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    
    plt.figure(figsize=(16, 4 * n_rows))
    
    for plot_idx, row in enumerate(df_to_display.to_dict("records"), start=1):
        
        class_name = row["class"]
        image_index = int(row["image_index"])
        
        img = class_image_arrays[class_name][image_index]
        
        if img.ndim == 3:
            img_display = img.mean(axis=2)
        else:
            img_display = img
        
        plt.subplot(n_rows, n_cols, plot_idx)
        plt.imshow(img_display, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        
        plt.title(
            f"{class_name} | idx={image_index}\n"
            f"mean={row['pixel_mean']:.1f} | std={row['pixel_std']:.1f}\n"
            f"dark={row['ratio_pixels_tres_sombres']:.3f} | "
            f"bright={row['ratio_pixels_tres_clairs']:.3f}",
            fontsize=9
        )
    
    if title is not None:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def display_grey_image_stats(stats_df, class_image_arrays, n=12, title="Suspicious images"):
    """
    Affiche les n premières images d'un DataFrame de statistiques.
    class_image_arrays: il faut que les images soit en gris
    """
    
    stats_df = stats_df.head(n).reset_index(drop=True)
    
    if len(stats_df) == 0:
        print("Aucune image à afficher.")
        return
    
    n_cols = 4
    n_rows = int(np.ceil(len(stats_df) / n_cols))
    
    plt.figure(figsize=(16, 4 * n_rows))
    plt.suptitle(title, fontsize=16)
    
    columns = [
        "class",
        "image_index",
        "pixel_mean",
        "pixel_std",
        "ratio_pixels_tres_sombres",
        "ratio_pixels_tres_clairs"
    ]
    
    for i, row in enumerate(
        stats_df[columns].itertuples(index=False, name=None),
        start=1
    ):
        class_name, image_index, pixel_mean, pixel_std, dark_ratio, bright_ratio = row
        
        img = class_image_arrays[class_name][int(image_index)]
        
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        plt.axis("off")
        
        plt.title(
            f"{class_name} | idx={image_index}\n"
            f"mean={pixel_mean:.1f} | std={pixel_std:.1f}\n"
            f"dark={dark_ratio:.3f} | bright={bright_ratio:.3f}",
            fontsize=9
        )
    
    plt.tight_layout()
    plt.show()

def clean_class_image_and_mask_arrays(class_image_arrays, class_mask_arrays, to_remove_df):
    """
    Crée une version nettoyée des images et des masks.
    
    Les images à supprimer sont définies dans to_remove_df avec :
    - class
    - image_index
    
    Important :
    On supprime les mêmes indices dans les images et dans les masks.
    """
    
    class_image_arrays_clean = {}
    class_mask_arrays_clean = {}
    
    remove_map = (
        to_remove_df
        .groupby("class")["image_index"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )
    
    for class_name, img_arr in class_image_arrays.items():
        
        mask_arr = class_mask_arrays[class_name]
        
        if img_arr.shape[0] != mask_arr.shape[0]:
            raise ValueError(
                f"Problème pour la classe {class_name} : "
                f"{img_arr.shape[0]} images mais {mask_arr.shape[0]} masks."
            )
        
        indices_to_remove = remove_map.get(class_name, set())
        
        indices_to_keep = [
            i for i in range(img_arr.shape[0])
            if i not in indices_to_remove
        ]
        
        class_image_arrays_clean[class_name] = img_arr[indices_to_keep]
        class_mask_arrays_clean[class_name] = mask_arr[indices_to_keep]
    
    return class_image_arrays_clean, class_mask_arrays_clean

def prepare_image_for_display(img):
    """
    Prépare une image pour l'affichage matplotlib.
    Gère RGB, grayscale, dtype object/int64, et valeurs 0-1 ou 0-255.
    """
    
    img = np.asarray(img)
    img = np.squeeze(img)
    
    # Si image RGB, conversion en niveaux de gris
    if img.ndim == 3:
        img = img.mean(axis=2)
    
    # Conversion dtype compatible matplotlib
    img = img.astype(np.float32)
    
    # Gestion de l'échelle
    if img.max() <= 1:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = 0, 255
    
    return img, vmin, vmax


def prepare_mask_for_display(mask):
    """
    Prépare un mask pour l'affichage.
    Convertit le mask en format 2D binaire 0/1.
    """
    
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    
    # Si mask RGB, conversion en niveaux de gris
    if mask.ndim == 3:
        mask = mask.mean(axis=2)
    
    mask = mask.astype(np.float32)
    
    # Conversion en mask binaire
    mask_binary = (mask > 0).astype(np.float32)
    
    return mask_binary


def display_image_mask_pair(class_image_arrays, class_mask_arrays, class_name, image_index):
    """
    Affiche une image, son mask associé, et une superposition des deux.
    Permet de vérifier que l'image correspond bien au mask.
    """
    
    if class_name not in class_image_arrays:
        raise KeyError(f"La classe '{class_name}' n'existe pas dans class_image_arrays.")
    
    if class_name not in class_mask_arrays:
        raise KeyError(f"La classe '{class_name}' n'existe pas dans class_mask_arrays.")
    
    if image_index >= class_image_arrays[class_name].shape[0]:
        raise IndexError(
            f"Index image invalide : {image_index}. "
            f"La classe '{class_name}' contient {class_image_arrays[class_name].shape[0]} images."
        )
    
    if image_index >= class_mask_arrays[class_name].shape[0]:
        raise IndexError(
            f"Index mask invalide : {image_index}. "
            f"La classe '{class_name}' contient {class_mask_arrays[class_name].shape[0]} masks."
        )
    
    img = class_image_arrays[class_name][image_index]
    mask = class_mask_arrays[class_name][image_index]
    
    img_display, vmin, vmax = prepare_image_for_display(img)
    mask_display = prepare_mask_for_display(mask)
    
    print("Classe :", class_name)
    print("Index :", image_index)
    print("Image shape :", np.asarray(img).shape, "| dtype :", np.asarray(img).dtype)
    print("Mask shape  :", np.asarray(mask).shape, "| dtype :", np.asarray(mask).dtype)
    print("Image min/max :", img_display.min(), img_display.max())
    print("Mask valeurs uniques :", np.unique(mask_display))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_display, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask")
    axes[1].axis("off")
    
    axes[2].imshow(img_display, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].imshow(mask_display, cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    axes[2].set_title("Image + mask")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
