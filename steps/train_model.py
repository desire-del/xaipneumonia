from zenml.steps import step, StepContext
from src.log import logger
import tensorflow as tf
from steps.preprocessing import preprocessing_step
from zenml.integrations.tensorflow.materializers.keras_materializer import KerasMaterializer
from src.base.config_entity import DataPreprocessingConfig

# @step(output_materializers={"model": KerasMaterializer}, enable_cache=False)
def train_model(
    train_dir: str,
    val_dir: str,
    data_preprocess_config: DataPreprocessingConfig
) -> tuple[tf.keras.Model, tf.keras.callbacks.History, dict]:
    """Train a VGG16 model using Functional API and log metrics to ZenML."""
    logger.info("Building VGG16 model...")

    # Prétraitement des données
    train_data, val_data,test_data, index_to_class = preprocessing_step(train_dir, val_dir, data_preprocess_config)

    # Définir l'input
    input_tensor = tf.keras.Input(shape=(256, 256, 3), name="input_image")

    # Charger VGG16 comme base
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=input_tensor
    )
    base_model.trainable = False

    # Ajout des couches de classification
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output_tensor = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # binaire

    # Créer le modèle final
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logger.info("Model built successfully.")
    model.summary()

    logger.info("Training model...")
    steps_per_epoch = len(train_data)
    validation_steps = len(val_data)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
        ]
    )

    # Log final metrics
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]

    logger.info(f"Final train accuracy: {final_train_acc}")
    logger.info(f"Final validation accuracy: {final_val_acc}")
    logger.info(f"Final train loss: {final_train_loss}")
    logger.info(f"Final validation loss: {final_val_loss}")

    return model, history, index_to_class
