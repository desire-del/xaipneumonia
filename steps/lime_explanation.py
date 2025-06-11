import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
from src.base.xai_entity import ImageExplainer
from .preprocess_input import preprocess_input


class LIMEExplainer(ImageExplainer):
    def __init__(self, model, input_shape=(256, 256, 3)):
        super().__init__(model)
        self.input_shape = input_shape
        self.explainer = lime_image.LimeImageExplainer()

    def explain(self, image_path_or_array, num_samples=1000):
        # Prétraitement de l'image
        if isinstance(image_path_or_array, str):
            image = preprocess_input(image_path_or_array)[0]
        elif isinstance(image_path_or_array, np.ndarray):
            image = image_path_or_array
        else:
            raise ValueError("image_path_or_array must be a path or a numpy array")

        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)

        image_batch = np.expand_dims(image_uint8, axis=0)  # (1, H, W, C)

        def predict_fn(images):
            images = tf.convert_to_tensor(images / 255.0, dtype=tf.float32)
            return self.model.predict(images, verbose=0)

        # Génération de l’explication
        print("image_uint8 shape:", image_uint8.shape)
        explanation = self.explainer.explain_instance(
            image_uint8,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )

        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=True,
            hide_rest=False,
            num_features=10,
            min_weight=0.0
        )

        superimposed = mark_boundaries(temp / 255.0, mask)

        return {
            "original_image": image_uint8,
            "pred_score": self.model.predict(image_batch),
            "lime_mask": mask,
            "superimposed_image": (superimposed * 255).astype(np.uint8)
        }
