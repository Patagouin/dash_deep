"""
Module Transformer pour la prédiction de séries temporelles financières.
Contient:
- Transformer avec attention multi-têtes
- Hybride LSTM + Transformer
- Utilitaires de téléchargement de modèles pré-entraînés
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
    Add, Embedding
)
from tensorflow.keras.optimizers import Adam
import logging
import hashlib
import requests
import tempfile
import os
from typing import Optional, Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO)


# ==============================================================================
# Couches personnalisées pour le Transformer
# ==============================================================================

class PositionalEncoding(layers.Layer):
    """
    Encodage positionnel pour le Transformer.
    Ajoute des informations de position aux embeddings d'entrée.
    """
    def __init__(self, max_len: int = 5000, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
    def build(self, input_shape):
        # Créer la matrice d'encodage positionnel
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        if self.d_model > 1:
            pe[:, 1::2] = np.cos(position * div_term[:self.d_model // 2])
        
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(self.max_len, self.d_model),
            initializer=tf.constant_initializer(pe),
            trainable=False
        )
        super().build(input_shape)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'd_model': self.d_model
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """
    Bloc encodeur Transformer avec:
    - Multi-Head Attention
    - Feed-Forward Network
    - Connexions résiduelles et LayerNorm
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dropout(dropout),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=None):
        # Self-attention avec connexion résiduelle
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward avec connexion résiduelle
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate
        })
        return config


# ==============================================================================
# Modèle Transformer pur
# ==============================================================================

def create_transformer_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    ff_multiplier: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return'
) -> Model:
    """
    Crée un modèle Transformer pour la prédiction de séries temporelles.
    
    Args:
        look_back: Taille de la fenêtre d'entrée
        num_features: Nombre de features par timestep
        nb_y: Nombre de points à prédire
        embed_dim: Dimension des embeddings
        num_heads: Nombre de têtes d'attention
        num_layers: Nombre de blocs encodeur
        ff_multiplier: Multiplicateur pour la dimension du FFN (ff_dim = embed_dim * ff_multiplier)
        dropout: Taux de dropout
        learning_rate: Taux d'apprentissage
        use_directional_accuracy: Utiliser la métrique DA
        prediction_type: 'return' ou 'price'
    
    Returns:
        Modèle Keras compilé
    """
    ff_dim = embed_dim * ff_multiplier
    
    inputs = Input(shape=(look_back, num_features))
    
    # Projection linéaire vers embed_dim
    x = Dense(embed_dim)(inputs)
    
    # Encodage positionnel
    x = PositionalEncoding(max_len=look_back, d_model=embed_dim)(x)
    x = Dropout(dropout)(x)
    
    # Blocs encodeur Transformer
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f'transformer_block_{i}'
        )(x)
    
    # Pooling global et sortie
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(embed_dim, activation='gelu')(x)
    x = Dropout(dropout / 2)(x)
    outputs = Dense(nb_y)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Métriques
    metrics_list = _build_metrics(use_directional_accuracy, prediction_type)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=metrics_list
    )
    
    logging.info(f"[Transformer] Modèle créé: look_back={look_back}, embed_dim={embed_dim}, "
                 f"heads={num_heads}, layers={num_layers}, params={model.count_params()}")
    
    return model


# ==============================================================================
# Modèle Hybride LSTM + Transformer
# ==============================================================================

def create_hybrid_lstm_transformer_model(
    look_back: int,
    num_features: int,
    nb_y: int,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    embed_dim: int = 64,
    num_heads: int = 4,
    transformer_layers: int = 1,
    ff_multiplier: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    fusion_mode: str = 'concat'
) -> Model:
    """
    Crée un modèle hybride combinant LSTM et Transformer.
    
    L'architecture utilise:
    - LSTM pour capturer les dépendances séquentielles locales
    - Transformer pour les relations globales via attention
    - Fusion des deux représentations
    
    Args:
        look_back: Taille de la fenêtre d'entrée
        num_features: Nombre de features par timestep
        nb_y: Nombre de points à prédire
        lstm_units: Unités LSTM
        lstm_layers: Nombre de couches LSTM
        embed_dim: Dimension des embeddings Transformer
        num_heads: Nombre de têtes d'attention
        transformer_layers: Nombre de blocs Transformer
        ff_multiplier: Multiplicateur FFN
        dropout: Taux de dropout
        learning_rate: Taux d'apprentissage
        use_directional_accuracy: Utiliser la métrique DA
        prediction_type: 'return' ou 'price'
        fusion_mode: 'concat', 'add', ou 'attention'
    
    Returns:
        Modèle Keras compilé
    """
    ff_dim = embed_dim * ff_multiplier
    
    inputs = Input(shape=(look_back, num_features))
    
    # ============ Branche LSTM ============
    lstm_x = inputs
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)  # Dernière couche ne retourne pas de séquence
        lstm_x = LSTM(
            lstm_units, 
            return_sequences=return_seq,
            dropout=dropout if return_seq else 0.0,
            name=f'lstm_{i}'
        )(lstm_x)
    
    # Si dernière LSTM ne retourne pas de séquence, on a shape (batch, lstm_units)
    # Sinon on prend le dernier état
    if lstm_layers > 0:
        lstm_out = lstm_x  # shape: (batch, lstm_units)
    else:
        lstm_out = Dense(lstm_units)(inputs[:, -1, :])
    
    # ============ Branche Transformer ============
    # Projection vers embed_dim
    trans_x = Dense(embed_dim)(inputs)
    trans_x = PositionalEncoding(max_len=look_back, d_model=embed_dim)(trans_x)
    trans_x = Dropout(dropout)(trans_x)
    
    for i in range(transformer_layers):
        trans_x = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f'hybrid_transformer_block_{i}'
        )(trans_x)
    
    # Pooling global pour obtenir une représentation fixe
    trans_out = GlobalAveragePooling1D()(trans_x)  # shape: (batch, embed_dim)
    
    # ============ Fusion ============
    if fusion_mode == 'concat':
        # Concaténation simple
        fused = Concatenate()([lstm_out, trans_out])
    elif fusion_mode == 'add':
        # Addition après projection à la même dimension
        if lstm_units != embed_dim:
            lstm_proj = Dense(max(lstm_units, embed_dim))(lstm_out)
            trans_proj = Dense(max(lstm_units, embed_dim))(trans_out)
            fused = Add()([lstm_proj, trans_proj])
        else:
            fused = Add()([lstm_out, trans_out])
    elif fusion_mode == 'attention':
        # Cross-attention entre LSTM et Transformer
        # Reshape lstm_out pour attention: (batch, 1, dim)
        lstm_reshaped = tf.expand_dims(lstm_out, axis=1)
        trans_reshaped = tf.expand_dims(trans_out, axis=1)
        
        cross_att = MultiHeadAttention(num_heads=2, key_dim=32, name='cross_attention')
        attended = cross_att(lstm_reshaped, trans_reshaped)
        attended = tf.squeeze(attended, axis=1)
        fused = Concatenate()([lstm_out, trans_out, attended])
    else:
        fused = Concatenate()([lstm_out, trans_out])
    
    # ============ Couches de sortie ============
    x = Dropout(dropout)(fused)
    x = Dense(embed_dim, activation='gelu')(x)
    x = Dropout(dropout / 2)(x)
    outputs = Dense(nb_y)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Métriques
    metrics_list = _build_metrics(use_directional_accuracy, prediction_type)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=metrics_list
    )
    
    logging.info(f"[Hybrid] Modèle créé: look_back={look_back}, lstm_units={lstm_units}, "
                 f"embed_dim={embed_dim}, fusion={fusion_mode}, params={model.count_params()}")
    
    return model


# ==============================================================================
# Utilitaires
# ==============================================================================

def _build_metrics(use_directional_accuracy: bool, prediction_type: str) -> list:
    """Construit la liste des métriques pour la compilation."""
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
    
    return metrics_list


def get_model_architecture_info(model_type: str) -> Dict[str, Any]:
    """
    Retourne les informations sur les paramètres disponibles pour chaque type de modèle.
    """
    architectures = {
        'lstm': {
            'name': 'LSTM',
            'description': 'Réseau récurrent Long Short-Term Memory',
            'params': ['units', 'layers', 'learning_rate', 'epochs'],
            'default_params': {
                'units': 64,
                'layers': 1,
                'learning_rate': 0.001,
                'epochs': 10
            }
        },
        'transformer': {
            'name': 'Transformer',
            'description': 'Transformer avec attention multi-têtes',
            'params': ['embed_dim', 'num_heads', 'num_layers', 'ff_multiplier', 'dropout', 'learning_rate', 'epochs'],
            'default_params': {
                'embed_dim': 64,
                'num_heads': 4,
                'num_layers': 2,
                'ff_multiplier': 4,
                'dropout': 0.1,
                'learning_rate': 0.001,
                'epochs': 10
            }
        },
        'hybrid': {
            'name': 'Hybride LSTM+Transformer',
            'description': 'Combinaison LSTM et Transformer avec fusion des représentations',
            'params': ['lstm_units', 'lstm_layers', 'embed_dim', 'num_heads', 'transformer_layers', 
                      'ff_multiplier', 'dropout', 'fusion_mode', 'learning_rate', 'epochs'],
            'default_params': {
                'lstm_units': 64,
                'lstm_layers': 1,
                'embed_dim': 64,
                'num_heads': 4,
                'transformer_layers': 1,
                'ff_multiplier': 4,
                'dropout': 0.1,
                'fusion_mode': 'concat',
                'learning_rate': 0.001,
                'epochs': 10
            }
        }
    }
    return architectures.get(model_type, architectures['lstm'])


# ==============================================================================
# Téléchargement de modèles pré-entraînés
# ==============================================================================

# Registre de modèles pré-entraînés disponibles
PRETRAINED_MODELS_REGISTRY = {
    'financial-transformer-small': {
        'url': None,  # À remplir avec une URL réelle si disponible
        'description': 'Petit Transformer pour séries financières',
        'architecture': 'transformer',
        'params': {
            'embed_dim': 32,
            'num_heads': 2,
            'num_layers': 2,
            'ff_multiplier': 4
        },
        'sha256': None
    },
    'hybrid-market-predictor': {
        'url': None,
        'description': 'Modèle hybride LSTM+Transformer pour marchés',
        'architecture': 'hybrid',
        'params': {
            'lstm_units': 64,
            'lstm_layers': 1,
            'embed_dim': 64,
            'num_heads': 4,
            'transformer_layers': 1,
            'fusion_mode': 'concat'
        },
        'sha256': None
    }
}


def list_available_pretrained_models() -> List[Dict[str, Any]]:
    """
    Liste les modèles pré-entraînés disponibles.
    
    Returns:
        Liste de dictionnaires avec les informations des modèles
    """
    result = []
    for name, info in PRETRAINED_MODELS_REGISTRY.items():
        result.append({
            'name': name,
            'description': info['description'],
            'architecture': info['architecture'],
            'available': info['url'] is not None
        })
    return result


def download_pretrained_model(
    model_name: str, 
    save_path: Optional[str] = None
) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """
    Télécharge un modèle pré-entraîné depuis une URL.
    
    Args:
        model_name: Nom du modèle dans le registre
        save_path: Chemin optionnel pour sauvegarder le modèle
        
    Returns:
        Tuple (bytes du modèle ou None, dict des métadonnées)
    """
    if model_name not in PRETRAINED_MODELS_REGISTRY:
        raise ValueError(f"Modèle '{model_name}' non trouvé dans le registre. "
                        f"Modèles disponibles: {list(PRETRAINED_MODELS_REGISTRY.keys())}")
    
    model_info = PRETRAINED_MODELS_REGISTRY[model_name]
    
    if model_info['url'] is None:
        logging.warning(f"[Pretrained] Modèle '{model_name}' n'a pas d'URL configurée")
        return None, model_info
    
    try:
        logging.info(f"[Pretrained] Téléchargement de '{model_name}' depuis {model_info['url']}")
        response = requests.get(model_info['url'], timeout=60)
        response.raise_for_status()
        
        model_bytes = response.content
        
        # Vérification du hash si disponible
        if model_info.get('sha256'):
            actual_hash = hashlib.sha256(model_bytes).hexdigest()
            if actual_hash != model_info['sha256']:
                raise ValueError(f"Hash mismatch: attendu {model_info['sha256']}, obtenu {actual_hash}")
        
        # Sauvegarde optionnelle
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(model_bytes)
            logging.info(f"[Pretrained] Modèle sauvegardé: {save_path}")
        
        return model_bytes, model_info
        
    except Exception as e:
        logging.error(f"[Pretrained] Erreur téléchargement: {e}")
        return None, model_info


def load_pretrained_from_bytes(
    model_bytes: bytes,
    custom_objects: Optional[Dict] = None
) -> Model:
    """
    Charge un modèle Keras depuis des bytes.
    
    Args:
        model_bytes: Bytes du fichier modèle
        custom_objects: Objets personnalisés pour le chargement
        
    Returns:
        Modèle Keras
    """
    # Objets personnalisés par défaut
    default_custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoderBlock': TransformerEncoderBlock,
    }
    
    if custom_objects:
        default_custom_objects.update(custom_objects)
    
    with tempfile.NamedTemporaryFile(suffix='.keras', delete=True) as tmp:
        tmp.write(model_bytes)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name, custom_objects=default_custom_objects, compile=False)
    
    return model


# ==============================================================================
# Fonction factory pour créer un modèle selon le type
# ==============================================================================

def create_model_by_type(
    model_type: str,
    look_back: int,
    num_features: int,
    nb_y: int,
    learning_rate: float = 0.001,
    use_directional_accuracy: bool = True,
    prediction_type: str = 'return',
    **kwargs
) -> Model:
    """
    Factory pour créer un modèle selon son type.
    
    Args:
        model_type: 'lstm', 'transformer', ou 'hybrid'
        look_back: Taille de la fenêtre d'entrée
        num_features: Nombre de features
        nb_y: Nombre de sorties
        learning_rate: Taux d'apprentissage
        use_directional_accuracy: Utiliser DA comme métrique
        prediction_type: 'return' ou 'price'
        **kwargs: Paramètres spécifiques au type de modèle
        
    Returns:
        Modèle Keras compilé
    """
    if model_type == 'transformer':
        return create_transformer_model(
            look_back=look_back,
            num_features=num_features,
            nb_y=nb_y,
            embed_dim=kwargs.get('embed_dim', 64),
            num_heads=kwargs.get('num_heads', 4),
            num_layers=kwargs.get('num_layers', 2),
            ff_multiplier=kwargs.get('ff_multiplier', 4),
            dropout=kwargs.get('dropout', 0.1),
            learning_rate=learning_rate,
            use_directional_accuracy=use_directional_accuracy,
            prediction_type=prediction_type
        )
    elif model_type == 'hybrid':
        return create_hybrid_lstm_transformer_model(
            look_back=look_back,
            num_features=num_features,
            nb_y=nb_y,
            lstm_units=kwargs.get('lstm_units', 64),
            lstm_layers=kwargs.get('lstm_layers', 1),
            embed_dim=kwargs.get('embed_dim', 64),
            num_heads=kwargs.get('num_heads', 4),
            transformer_layers=kwargs.get('transformer_layers', 1),
            ff_multiplier=kwargs.get('ff_multiplier', 4),
            dropout=kwargs.get('dropout', 0.1),
            learning_rate=learning_rate,
            use_directional_accuracy=use_directional_accuracy,
            prediction_type=prediction_type,
            fusion_mode=kwargs.get('fusion_mode', 'concat')
        )
    else:
        # LSTM par défaut (utilise la fonction existante dans playground)
        raise ValueError(f"Pour le type 'lstm', utilisez _build_lstm_model de playground.py. "
                        f"Ce factory supporte: transformer, hybrid")


# ==============================================================================
# Export des objets personnalisés pour le chargement
# ==============================================================================

def get_custom_objects() -> Dict[str, Any]:
    """
    Retourne le dictionnaire des objets personnalisés nécessaires
    pour charger des modèles Transformer/Hybrid sauvegardés.
    """
    return {
        'PositionalEncoding': PositionalEncoding,
        'TransformerEncoderBlock': TransformerEncoderBlock,
    }



