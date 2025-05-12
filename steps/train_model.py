from zenml.steps import step, StepContext
from src.log import logger
import tensorflow as tf
from zenml.integrations.tensorflow.materializers.keras_materializer import KerasModelMaterializer

@step(
    output_materializers={"model": KerasModelMaterializer},
    enable_cache=False
)
def train_model(
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    context: StepContext
) -> tf.keras.Model:
    """Train a VGG16 model and log metrics to ZenML."""
    logger.info("Building VGG16 model...")

    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(180, 180, 3)
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Training model...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5,
        verbose=1
    )

    # Log final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    logger.info(f"Final train accuracy: {final_train_acc}")
    logger.info(f"Final validation accuracy: {final_val_acc}")

    # Log to experiment tracker if available
    if context.stack.experiment_tracker:
        context.stack.experiment_tracker.log_metrics({
            "train_accuracy": final_train_acc,
            "val_accuracy": final_val_acc,
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
        })

    return model
