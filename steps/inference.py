
import tensorflow as tf
import numpy as np

def inference(model, image_input, target_size=(256, 256), normalize=True):
    """
    Effectue l'inférence d'une image à partir d'un modèle Keras.

    Args:
        model: modèle TensorFlow/Keras chargé.
        image_input (str | np.ndarray): chemin vers l'image ou image déjà chargée.
        target_size (tuple): taille d'entrée du modèle (ex: (256, 256)).
        normalize (bool): si True, normalise l'image entre 0 et 1.

    Returns:
        tuple:
            - image prétraitée (shape: (1, H, W, 3))
            - prédiction
    """
    if isinstance(image_input, str):
        # Chargement depuis le chemin
        img = tf.keras.utils.load_img(image_input, target_size=target_size)
        img_array = tf.keras.utils.img_to_array(img)
    elif isinstance(image_input, np.ndarray):
        # Image déjà chargée (on suppose que c’est HxWxC)
        img_array = image_input
        if img_array.shape[:2] != target_size:
            img_array = tf.image.resize(img_array, target_size).numpy()
    else:
        raise ValueError("image_input must be a file path or a numpy array.")

    # Ajout de la dimension batch
    img_array = tf.expand_dims(img_array, axis=0)

    if normalize:
        img_array = img_array / 255.0

    predictions = model.predict(img_array)
    return img_array, predictions
