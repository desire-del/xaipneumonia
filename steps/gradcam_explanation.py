import tensorflow as tf
import numpy as np
import cv2
import matplotlib as mpl
from src.base.xai_entity import ImageExplainer
from .preprocess_input import preprocess_input
from tensorflow.keras.utils import array_to_img, img_to_array
from skimage.transform import resize

class GradCAMExplainer(ImageExplainer):
    def __init__(self, model, last_conv_layer_name="block5_conv3", input_shape=(256, 256, 3)):
        super().__init__(model)
        self.last_conv_layer_name = last_conv_layer_name
        self.input_shape = input_shape

    def get_gradcam_heatmap(self, img_array, pred_index=None):
        grad_model = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
            class_output = predictions[:, pred_index]

        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # Shape (H, W, C)

        # Compute weighted sum over channels (correct Grad-CAM)
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Normalize between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy(), predictions

    def explain(self, image_path_or_array, alpha=0.4, color_map="jet"):
        # Load and preprocess image
        if isinstance(image_path_or_array, str):
            img_array = preprocess_input(image_path_or_array)  # Should return shape (1, H, W, 3)
        elif isinstance(image_path_or_array, np.ndarray):
            if len(image_path_or_array.shape) == 3:  # Single image (H, W, C)
                img_array = np.expand_dims(image_path_or_array, axis=0)
            elif len(image_path_or_array.shape) == 4:  # Batch of images (N, H, W, C)
                img_array = image_path_or_array
            else:
                raise ValueError("Input array must have shape (H,W,C) or (N,H,W,C)")
            
            # Ensure proper preprocessing if needed
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
        else:
            raise ValueError("Input must be either a path string or numpy array")

        input_img = img_array[0]  # Remove batch dimension for visualization

        # Get Grad-CAM heatmap and prediction
        heatmap, predictions = self.get_gradcam_heatmap(img_array)
        heatmap_uint8 = np.uint8(255 * heatmap)

        # Apply colormap using matplotlib
        cmap = mpl.colormaps[color_map]
        colors = cmap(np.arange(256))[:, :3]  # Get RGB values
        heatmap_color = colors[heatmap_uint8]  # Map heatmap to RGB values (float32 0-1)
        heatmap_color = np.uint8(heatmap_color * 255)  # Convert to uint8

        # Resize heatmap to match image size
        heatmap_color = cv2.resize(heatmap_color, (input_img.shape[1], input_img.shape[0]))

        # Convert heatmap to float32 for blending
        heatmap_color = heatmap_color.astype(np.float32)
        
        # Ensure input image is in 0-255 range
        if input_img.max() <= 1.0:
            input_img = (input_img * 255).astype(np.uint8)
        else:
            input_img = input_img.astype(np.uint8)

        # Blend heatmap with original image
        superimposed_img = cv2.addWeighted(input_img.astype(np.float32), 1-alpha, heatmap_color, alpha, 0)
        superimposed_img = np.uint8(superimposed_img)
        target_shape = (256, 256)  # Resize to target shape if needed
        resized_heatmap = resize(heatmap, target_shape, order=1, mode='reflect', preserve_range=True)

        return {
            "original_image": input_img,
            "pred_score": predictions.numpy(),
            "heatmap": resized_heatmap,
            "superimposed_image": superimposed_img
        }