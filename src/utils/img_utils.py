import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import List, Tuple, Optional, Union
from PIL import Image

class FeatureMapUtils:
    """
    Utility class for working with feature maps in CNN-based models using TensorFlow/Keras.
    
    Features include:
    - Visualizing feature maps
    - Extracting feature maps from specific layers
    - Analyzing feature map statistics
    - Creating feature map grids
    - Overlaying feature maps on original images
    """
    
    @staticmethod
    def get_feature_maps(model: tf.keras.Model, 
                        input_tensor: tf.Tensor, 
                        target_layers: List[Union[str, tf.keras.layers.Layer]]) -> dict:
        """
        Extract feature maps from specified layers of a CNN model.
        
        Args:
            model: The CNN model (tf.keras.Model)
            input_tensor: Input tensor (batch_size, height, width, channels)
            target_layers: List of layer names or layer objects to extract features from
            
        Returns:
            Dictionary with layer names as keys and feature maps as values
        """
        # Convert layer objects to their names
        layer_names = []
        for layer in target_layers:
            if isinstance(layer, str):
                layer_names.append(layer)
            else:
                layer_names.append(layer.name)
        
        # Create a submodel that outputs the feature maps from target layers
        outputs = [model.get_layer(name).output for name in layer_names]
        feature_extractor = tf.keras.Model(inputs=model.input, outputs=outputs)
        
        # Get feature maps
        feature_maps = feature_extractor(input_tensor)
        
        # Convert single output to list if needed
        if not isinstance(feature_maps, list):
            feature_maps = [feature_maps]
        
        return {name: fm for name, fm in zip(layer_names, feature_maps)}
    
    @staticmethod
    def visualize_feature_maps(feature_maps: tf.Tensor, 
                            num_maps: Optional[int] = None,
                            figsize: Tuple[int, int] = (10, 10),
                            cmap: str = 'viridis',
                            max_channels: int = 64) -> None:
        """
        Improved version that handles high-channel feature maps.
        
        Args:
            feature_maps: Tensor of shape (batch_size, height, width, channels)
            num_maps: Number of feature maps to display (None for all up to max_channels)
            figsize: Size of the figure
            cmap: Colormap to use
            max_channels: Maximum number of channels to display
        """
        # Convert to numpy if needed and take first batch element
        if tf.is_tensor(feature_maps):
            feature_maps = feature_maps.numpy()
        feature_maps = feature_maps[0]  # Shape: (height, width, channels)
        
        # Determine how many feature maps to show
        num_channels = feature_maps.shape[-1]
        if num_maps is None:
            num_maps = min(num_channels, max_channels)
        else:
            num_maps = min(num_maps, num_channels, max_channels)
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_maps)))
        rows = int(np.ceil(num_maps / cols))
        
        plt.figure(figsize=figsize)
        for i in range(num_maps):
            plt.subplot(rows, cols, i+1)
            
            # Get single channel and normalize
            channel_data = feature_maps[..., i]
            channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
            
            plt.imshow(channel_data, cmap=cmap)
            plt.axis('off')
            plt.title(f'FuturMap {i}')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_feature_grid(feature_maps: tf.Tensor, 
                          normalize: bool = True) -> np.ndarray:
        """
        Create a grid of feature maps for visualization.
        
        Args:
            feature_maps: Tensor of shape (batch_size, height, width, channels)
            normalize: Whether to normalize each feature map
            
        Returns:
            Numpy array representing the feature grid
        """
        # Convert to numpy if needed and take first batch element
        if tf.is_tensor(feature_maps):
            feature_maps = feature_maps.numpy()
        feature_maps = feature_maps[0]
        
        # Transpose to (channels, height, width) format
        if len(feature_maps.shape) == 4:
            feature_maps = np.transpose(feature_maps, (2, 0, 1))
        elif len(feature_maps.shape) == 3:
            feature_maps = np.expand_dims(feature_maps, axis=0)
        
        num_maps = feature_maps.shape[0]
        
        # Calculate grid dimensions
        cols = int(np.ceil(np.sqrt(num_maps)))
        rows = int(np.ceil(num_maps / cols))
        
        # Create empty grid
        grid_h = rows * feature_maps.shape[1]
        grid_w = cols * feature_maps.shape[2]
        grid = np.zeros((grid_h, grid_w))
        
        # Fill the grid
        for i in range(num_maps):
            r = i // cols
            c = i % cols
            h_start = r * feature_maps.shape[1]
            h_end = h_start + feature_maps.shape[1]
            w_start = c * feature_maps.shape[2]
            w_end = w_start + feature_maps.shape[2]
            
            fm = feature_maps[i]
            if normalize:
                fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                
            grid[h_start:h_end, w_start:w_end] = fm
            
        return grid
    
    @staticmethod
    def analyze_feature_maps(feature_maps: tf.Tensor) -> dict:
        """
        Compute statistics about feature maps.
        
        Args:
            feature_maps: Tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'mean': tf.reduce_mean(feature_maps).numpy(),
            'std': tf.math.reduce_std(feature_maps).numpy(),
            'min': tf.reduce_min(feature_maps).numpy(),
            'max': tf.reduce_max(feature_maps).numpy(),
            'shape': feature_maps.shape.as_list(),
            'nonzero_percentage': (tf.math.count_nonzero(feature_maps).numpy() / 
                                tf.size(feature_maps).numpy()) * 100
        }
        return stats
    
    @staticmethod
    def overlay_feature_map(image: Union[np.ndarray, Image.Image],
                          feature_map: np.ndarray,
                          alpha: float = 0.5,
                          cmap: str = 'jet') -> np.ndarray:
        """
        Overlay a feature map on top of an original image.
        
        Args:
            image: Original image (H, W, C) or PIL Image
            feature_map: Feature map to overlay (H, W)
            alpha: Transparency of the overlay
            cmap: Colormap to use for the feature map
            
        Returns:
            Numpy array of the overlayed image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Normalize feature map
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Resize feature map to match image dimensions if needed
        if feature_map.shape != image.shape[:2]:
            from skimage.transform import resize
            feature_map = resize(feature_map, image.shape[:2])
            
        # Apply colormap to feature map
        cmap = plt.get_cmap(cmap)
        colored_feature_map = cmap(feature_map)[..., :3]  # Ignore alpha channel
        
        # Convert image to float and normalize
        image = image.astype(np.float32) / 255.0
        
        # Overlay
        overlayed = (1 - alpha) * image + alpha * colored_feature_map
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    @staticmethod
    def visualize_layer_activations(model: tf.keras.Model,
                                  input_tensor: tf.Tensor,
                                  layer_name: str,
                                  num_filters: int = 16,
                                  figsize: Tuple[int, int] = (10, 10)) -> None:
        """
        Visualize activations for a specific layer.
        
        Args:
            model: The CNN model
            input_tensor: Input tensor
            layer_name: Name of layer to visualize
            num_filters: Number of filters to display
            figsize: Figure size
        """
        # Create a model that outputs the activations of the target layer
        layer_output = model.get_layer(layer_name).output
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_output)
        
        # Get activations
        activations = activation_model(input_tensor)
        
        # Visualize
        FeatureMapUtils.visualize_feature_maps(activations, num_maps=num_filters, figsize=figsize)