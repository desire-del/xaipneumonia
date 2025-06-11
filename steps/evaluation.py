from zenml.steps import step
import tensorflow as tf
from typing import Tuple, Dict, Any
from src.log import logger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
#from zenml.artifacts import DataArtifact
#from zenml.metadata import MetadataTracker

#@step(enable_cache=False)
def evaluate_model(
    model: tf.keras.Model,
    val_data: tf.data.Dataset,
    index_to_class: Dict[int, str],
    #context: StepContext
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Evaluate the trained model on validation data and log metrics.
    
    Args:
        model: Trained Keras model
        val_data: Validation dataset
        index_to_class: Mapping from class indices to class names
        context: ZenML step context for artifact logging
        
    Returns:
        Tuple containing (val_loss, val_accuracy, evaluation_metrics)
    """
    logger.info("Evaluating model on validation data...")
    
    # 1. Basic evaluation with model.evaluate()
    results = model.evaluate(val_data, return_dict=True)
    val_loss = results['loss']
    val_accuracy = results['accuracy']
    
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # 2. Get predictions in batch
    y_true = val_data.labels
    y_pred_probs = model.predict(val_data).squeeze()

    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # 3. Generate classification report
    class_names = list(index_to_class.values())
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # 4. Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Create and save visualizations
    with tempfile.TemporaryDirectory() as tempdir:
        # Confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = f"{tempdir}/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        
        # ROC curve (for binary classification)
        if len(class_names) == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            roc_path = f"{tempdir}/roc_curve.png"
            plt.savefig(roc_path)
            plt.close()
        else:
            roc_path = None
    
    # 6. Log artifacts to ZenML

    """
    tracker = MetadataTracker(context)
    
    # Log confusion matrix
    tracker.log_artifact(
        DataArtifact(
            name="confusion_matrix",
            uri=cm_path,
            metadata={"type": "image/png"}
        )
    )
    
    # Log ROC curve if binary classification
    if roc_path:
        tracker.log_artifact(
            DataArtifact(
                name="roc_curve",
                uri=roc_path,
                metadata={"type": "image/png"}
            )
        )
    """
    # 7. Prepare metrics dictionary
    evaluation_metrics = {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
    }
    
    # Add ROC AUC if binary classification
    if len(class_names) == 2:
        evaluation_metrics['roc_auc'] = roc_auc
    
    logger.info("Evaluation completed successfully.")
    
    return val_loss, val_accuracy, evaluation_metrics