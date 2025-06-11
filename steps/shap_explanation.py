import numpy as np
import shap
import tensorflow as tf
from src.base.xai_entity import ImageExplainer
from .preprocess_input import preprocess_input


class SHAPExplainer(ImageExplainer):
    def __init__(self, model, input_shape=(256, 256, 3), class_names=['PNEUMONIA', 'NORMAL']):
        super().__init__(model)
        self.input_shape = input_shape
        
        # Create a masker for SHAP - need to pass the full shape (H, W, C)
        self.masker = shap.maskers.Image("blur(128,128)", input_shape)
        self.class_names = class_names
        # Create SHAP explainer
        self.explainer = shap.Explainer(
            self._binary_predict,
            self.masker,
            output_names=self.class_names
        )
        self.shap_values = None
    def _binary_predict(self, images):
        """Wrapper function to format predictions for binary classification"""
        if isinstance(images, list):
            images = np.array(images)
        
        # Convert to float32 if needed and normalize
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
            
        preds = self.model.predict(images, verbose=0)
        
        return np.hstack([1-preds, preds])
    
    def explain(self, image_path_or_array, num_samples=100):
        """
        Explain the model's prediction using SHAP values.
        
        Args:
            image_path_or_array: Either a path to an image or a numpy array
            num_samples: Number of evaluations to approximate SHAP values
            
        Returns:
            Dictionary containing explanation results
        """
        # Preprocess the image
        if isinstance(image_path_or_array, str):
            image = preprocess_input(image_path_or_array)[0]
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise ValueError("image_path_or_array must be a path or a numpy array")

        # Ensure image is in 0-255 range for SHAP
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        # Expand dimensions for batch prediction
        image_batch = np.expand_dims(image_uint8, axis=0)
        
        # Get SHAP values
        self.shap_values = self.explainer(
            image_batch,
            max_evals=num_samples,
            batch_size=32,
            outputs=shap.Explanation.argsort.flip[:2]
        )
        
        # Get the predicted class
        pred_scores = self.model.predict(image_batch)
        pred_class = self.class_names[int(pred_scores[0][0] > 0.5)]
        
        
        return {
            "original_image": image_uint8,
            "pred_score": pred_scores,
            "shap_values": self.shap_values,
            "predicted_class": pred_class
        }
    
    def plot_shap_values(self):
        """
        Plot the SHAP values for the image.
        
        Args:
            shap_values: SHAP values to plot
            
        Returns:
            Matplotlib figure with SHAP values
        """
        return shap.plots.image(self.shap_values, show=False)