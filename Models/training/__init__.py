# -*- coding: utf-8 -*-
"""
Models.training - Logique d'entraînement ML réutilisable

Ce module contient les fonctions de préparation de données et de construction
de modèles utilisables dans toutes les pages (playground, prediction, etc.)
"""

from .data_preparation import (
    prepare_xy_from_store,
    prepare_xy_for_inference,
)

from .model_builders import (
    build_lstm_model,
)

from .callbacks import (
    TrainingProgressCallback,
    KerasProgressCallback,
    DashProgressCallback,  # Alias pour compatibilité
)

__all__ = [
    # Préparation données
    'prepare_xy_from_store',
    'prepare_xy_for_inference',
    # Construction modèles
    'build_lstm_model',
    # Callbacks
    'TrainingProgressCallback',
    'KerasProgressCallback',
    'DashProgressCallback',
]

