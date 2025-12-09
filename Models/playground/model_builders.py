"""
Fonctions de construction des modèles pour le Playground.
"""

import logging
from typing import Optional

# Importer TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("[ModelBuilders] TensorFlow non disponible")

# Importer les modèles Transformer
try:
    from Models.transformer import (
        create_transformer_model,
        create_hybrid_lstm_transformer_model,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logging.warning("[ModelBuilders] Module transformer non disponible")


# Valeurs par défaut
DEFAULT_UNITS = 64
DEFAULT_LAYERS = 1
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_TRANSFORMER_LAYERS = 2
DEFAULT_FF_MULTIPLIER = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_FUSION_MODE = 'concat'


def is_transformer_available() -> bool:
    """Retourne True si le module Transformer est disponible."""
    return TRANSFORMER_AVAILABLE


def build_lstm_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    units: int = DEFAULT_UNITS,
    layers: int = DEFAULT_LAYERS,
    lr: float = DEFAULT_LEARNING_RATE,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    loss_type: str = 'mse'
) -> 'tf.keras.Model':
    """
    Construit un modèle LSTM.
    
    Args:
        look_back: Taille de la fenêtre d'entrée
        num_features: Nombre de features
        nb_y: Nombre de sorties
        units: Nombre d'unités par couche
        layers: Nombre de couches
        lr: Learning rate
        use_directional_accuracy: Activer la métrique DA
        prediction_type: Type de prédiction
        loss_type: Type de loss ('mse', 'mae', 'scaled_mse')
    
    Returns:
        Modèle Keras compilé
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow n'est pas disponible")
    
    # Métrique de Directional Accuracy
    metrics_list = []
    if use_directional_accuracy:
        if prediction_type == 'price':
            def directional_accuracy_metric(y_true, y_pred):
                true_dir = tf.sign(y_true - 1.0)
                pred_dir = tf.sign(y_pred - 1.0)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        else:
            def directional_accuracy_metric(y_true, y_pred):
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
    loss_fn = 'mae' if loss_type == 'mae' else 'mse'
    
    inputs = tf.keras.Input(shape=(int(look_back), int(num_features)))
    x = inputs
    
    for i in range(int(max(1, layers))):
        return_seq = (i != int(layers) - 1)
        x = tf.keras.layers.LSTM(int(units), return_sequences=return_seq, dropout=0.0)(x)
    
    if prediction_type == 'signal':
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
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


def build_transformer_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS,
    ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LEARNING_RATE,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return'
) -> 'tf.keras.Model':
    """
    Construit un modèle Transformer.
    """
    if not TRANSFORMER_AVAILABLE:
        raise ImportError("Module transformer non disponible")
    
    return create_transformer_model(
        look_back, num_features, nb_y,
        embed_dim, num_heads, transformer_layers, ff_multiplier, dropout,
        lr, use_directional_accuracy, prediction_type
    )


def build_hybrid_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    lstm_units: int = DEFAULT_UNITS,
    lstm_layers: int = DEFAULT_LAYERS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    trans_layers: int = 1,
    ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
    dropout: float = DEFAULT_DROPOUT,
    lr: float = DEFAULT_LEARNING_RATE,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    fusion_mode: str = DEFAULT_FUSION_MODE
) -> 'tf.keras.Model':
    """
    Construit un modèle Hybride LSTM+Transformer.
    """
    if not TRANSFORMER_AVAILABLE:
        raise ImportError("Module transformer non disponible")
    
    return create_hybrid_lstm_transformer_model(
        look_back, num_features, nb_y,
        lstm_units, lstm_layers, embed_dim, num_heads, trans_layers,
        ff_multiplier, dropout, lr, use_directional_accuracy, prediction_type, fusion_mode
    )


def build_model_by_type(
    model_type: str,
    look_back: int,
    num_features: int,
    nb_y: int,
    units: int = DEFAULT_UNITS,
    layers: int = DEFAULT_LAYERS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    num_heads: int = DEFAULT_NUM_HEADS,
    transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS,
    ff_multiplier: int = DEFAULT_FF_MULTIPLIER,
    dropout: float = DEFAULT_DROPOUT,
    hybrid_lstm_units: Optional[int] = None,
    hybrid_lstm_layers: Optional[int] = None,
    hybrid_embed_dim: Optional[int] = None,
    hybrid_num_heads: Optional[int] = None,
    hybrid_trans_layers: Optional[int] = None,
    fusion_mode: str = DEFAULT_FUSION_MODE,
    hybrid_dropout: Optional[float] = None,
    lr: float = DEFAULT_LEARNING_RATE,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    loss_type: str = 'mse'
) -> 'tf.keras.Model':
    """
    Construit un modèle selon son type.
    
    Args:
        model_type: Type de modèle ('lstm', 'transformer', 'hybrid')
        ... autres paramètres ...
    
    Returns:
        Modèle Keras compilé
    """
    if model_type == 'transformer':
        if not TRANSFORMER_AVAILABLE:
            logging.warning("[ModelBuilders] Transformer non disponible, fallback vers LSTM")
            return build_lstm_model(
                look_back, num_features, nb_y, units, layers, lr,
                use_directional_accuracy, prediction_type, loss_type
            )
        return build_transformer_model(
            look_back, num_features, nb_y,
            embed_dim, num_heads, transformer_layers, ff_multiplier, dropout,
            lr, use_directional_accuracy, prediction_type
        )
    
    elif model_type == 'hybrid':
        if not TRANSFORMER_AVAILABLE:
            logging.warning("[ModelBuilders] Hybride non disponible, fallback vers LSTM")
            return build_lstm_model(
                look_back, num_features, nb_y, units, layers, lr,
                use_directional_accuracy, prediction_type, loss_type
            )
        return build_hybrid_model(
            look_back, num_features, nb_y,
            hybrid_lstm_units or units,
            hybrid_lstm_layers or layers,
            hybrid_embed_dim or embed_dim,
            hybrid_num_heads or num_heads,
            hybrid_trans_layers or 1,
            ff_multiplier,
            hybrid_dropout if hybrid_dropout is not None else dropout,
            lr, use_directional_accuracy, prediction_type, fusion_mode
        )
    
    else:  # lstm par défaut
        return build_lstm_model(
            look_back, num_features, nb_y, units, layers, lr,
            use_directional_accuracy, prediction_type, loss_type
        )

