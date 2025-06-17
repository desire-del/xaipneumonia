import numpy as np
import shap
import tensorflow as tf
from src.base.xai_entity import ImageExplainer
from steps.preprocess_input import preprocess_input
from src.log import logger


class GradientSHAPExplainer(ImageExplainer):
    def __init__(self, model, background_data, input_shape=(256, 256, 3), class_names=['PNEUMONIA', 'NORMAL']):
        super().__init__(model)
        self.input_shape = input_shape
        self.class_names = class_names

        # Convert background data to normalized float32 if needed
        if background_data.dtype != np.float32:
            background_data = background_data.astype(np.float32) / 255.0

        # Take a small representative sample for the background
        self.background = background_data[:100]  # shape: (100, H, W, C)

        self.explainer = shap.GradientExplainer(self.model, self.background)

    def explain(self, image_path_or_array, nsamples=50):
        # Load and preprocess image
        if isinstance(image_path_or_array, str):
            image = preprocess_input(image_path_or_array)[0]
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise ValueError("image_path_or_array must be a path or numpy array")

        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0

        image_batch = np.expand_dims(image, axis=0)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(image_batch, nsamples=nsamples)
        
        # For binary classification with sigmoid, shap_values is a list with 1 element
        if isinstance(shap_values, list) and len(shap_values) == 1:
            shap_values = shap_values[0]  
            
        pred_score = self.model.predict(image_batch, verbose=0)
        pred_class = self.class_names[int(pred_score[0][0] > 0.5)]

        logger.info(f"Predicted class: {pred_class}, Score: {pred_score[0][0]}")
        logger.info(f"SHAP values shape: {shap_values.shape}")

        return {
            "original_image": (image * 255).astype(np.uint8),
            "pred_score": pred_score,
            "shap_values": shap_values,
            "predicted_class": pred_class
        }
