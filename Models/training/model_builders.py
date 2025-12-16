# -*- coding: utf-8 -*-
"""
Fonctions de construction de modèles ML.
Extraites de playground.py pour réutilisation dans d'autres pages.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Pour les type hints uniquement, import normal
    import tensorflow as tf

# IMPORTANT: Import lazy de TensorFlow pour permettre la configuration CUDA
# dans les workers spawn avant l'import
def _get_tf():
    """Import lazy de TensorFlow."""
    import tensorflow as tf
    return tf


def build_lstm_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    units: int,
    layers: int,
    lr: float,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    loss_type: str = 'mse'
):
    """
    Construit un modèle LSTM.
    
    Args:
        look_back: Taille de la fenêtre d'entrée
        num_features: Nombre de features par point
        nb_y: Nombre de sorties (points à prédire)
        units: Nombre de neurones par couche LSTM
        layers: Nombre de couches LSTM empilées
        lr: Learning rate
        use_directional_accuracy: Activer la métrique DA
        prediction_type: 'return', 'price', ou 'signal'
        loss_type: 'mse', 'scaled_mse', ou 'mae'
    
    Returns:
        Modèle Keras compilé
    """
    # Import lazy de TensorFlow (après configuration CUDA dans le worker)
    tf = _get_tf()
    
    # Métrique de Directional Accuracy (DA)
    metrics_list = []
    
    if use_directional_accuracy:
        if prediction_type == 'price':
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur Prix normalisés: compare si le prix va au-dessus/en-dessous du prix de référence (1.0)
                true_dir = tf.sign(y_true - 1.0)
                pred_dir = tf.sign(y_pred - 1.0)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        else:
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur retours: compare les signes des variations
                true_dir = tf.sign(y_true)
                pred_dir = tf.sign(y_pred)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        
        try:
            directional_accuracy_metric.__name__ = 'directional_accuracy'
        except Exception:
            pass
        metrics_list.append(directional_accuracy_metric)
    
    # Choix de la loss
    if loss_type == 'mae':
        loss_fn = 'mae'
    else:
        loss_fn = 'mse'  # mse et scaled_mse utilisent tous deux mse (scaling fait sur les données)
    
    inputs = tf.keras.Input(shape=(int(look_back), int(num_features)))
    x = inputs
    
    for i in range(int(max(1, layers))):
        return_seq = (i != int(layers) - 1)
        x = tf.keras.layers.LSTM(int(units), return_sequences=return_seq, dropout=0.0)(x)
    
    if prediction_type == 'signal':
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        # Force loss/metrics pour signal
        loss_fn = 'sparse_categorical_crossentropy'
        metrics_list = ['accuracy']
    else:
        outputs = tf.keras.layers.Dense(int(nb_y))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss=loss_fn,
        metrics=metrics_list
    )
    return model

