"""
Module Models/playground - Logique calculatoire pour le Playground.

Ce module contient toute la logique de calcul (ML, préparation données, etc.)
utilisée par le Playground. La vue reste dans web/apps/playground.py.

Usage:
    from Models.playground import (
        prepare_xy_from_store,
        build_model_by_type,
        build_segments_graph,
    )
"""

from .data_preparation import (
    prepare_xy_from_store,
    prepare_xy_for_inference,
    estimate_nb_quotes_per_day,
)

from .model_builders import (
    build_lstm_model,
    build_transformer_model,
    build_hybrid_model,
    build_model_by_type,
    is_transformer_available,
)

from .graph_builders import (
    build_segments_graph,
    build_generalization_figure,
    build_training_history_figure,
    build_trades_table,
)

__all__ = [
    # Data preparation
    'prepare_xy_from_store',
    'prepare_xy_for_inference',
    'estimate_nb_quotes_per_day',
    # Model builders
    'build_lstm_model',
    'build_transformer_model',
    'build_hybrid_model',
    'build_model_by_type',
    'is_transformer_available',
    # Graph builders
    'build_segments_graph',
    'build_generalization_figure',
    'build_training_history_figure',
    'build_trades_table',
]

