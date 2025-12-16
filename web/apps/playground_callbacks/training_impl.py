# -*- coding: utf-8 -*-
"""
Implémentation "core" d'entraînement pour usage hors callbacks Dash.

Historique:
- Un ancien code faisait référence à `_train_model_impl` dans `playground.py`.
- Le symbole n'existait plus: on l'a réintroduit ici, au sein de `playground_callbacks/`.
"""

import logging
import tempfile

import numpy as np

from Models.training import build_lstm_model, prepare_xy_from_store
from web.apps.model_config import (
    DEFAULT_EPOCHS,
    DEFAULT_FF_MULTIPLIER,
    DEFAULT_FUSION_MODE,
    DEFAULT_LOOK_BACK,
    DEFAULT_NB_Y,
    DEFAULT_NUM_HEADS,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_FIRST_MINUTES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LSTM_LAYERS as DEFAULT_LAYERS,
    DEFAULT_LSTM_UNITS as DEFAULT_UNITS,
    DEFAULT_STRIDE,
    DEFAULT_TRANSFORMER_LAYERS,
)

try:
    from Models.transformer import (
        create_transformer_model,
        create_hybrid_lstm_transformer_model,
    )
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False


def _train_model_impl(
    store_json,
    look_back=DEFAULT_LOOK_BACK,
    stride=DEFAULT_STRIDE,
    nb_y=DEFAULT_NB_Y,
    first_minutes=DEFAULT_FIRST_MINUTES,
    use_directional_accuracy=True,
    loss_type='mse',
    units=DEFAULT_UNITS,
    layers=DEFAULT_LAYERS,
    lr=DEFAULT_LEARNING_RATE,
    epochs=DEFAULT_EPOCHS,
    prediction_type='return',
    model_type='lstm',
    embed_dim=DEFAULT_EMBED_DIM,
    num_heads=DEFAULT_NUM_HEADS,
    transformer_layers=DEFAULT_TRANSFORMER_LAYERS,
    ff_multiplier=DEFAULT_FF_MULTIPLIER,
    dropout=DEFAULT_DROPOUT,
    hybrid_lstm_units=DEFAULT_UNITS,
    hybrid_lstm_layers=DEFAULT_LAYERS,
    hybrid_embed_dim=DEFAULT_EMBED_DIM,
    hybrid_num_heads=DEFAULT_NUM_HEADS,
    hybrid_trans_layers=1,
    fusion_mode=DEFAULT_FUSION_MODE,
    hybrid_dropout=DEFAULT_DROPOUT,
    use_gpu=True,
):
    """
    Entraîne un modèle sur les données du store (JSON) et retourne un résultat sérialisable.

    Retour:
    - dict: { model_path, predictions_data, meta }
    """
    from Models.training.tf_worker_setup import setup_cuda_for_worker

    setup_cuda_for_worker(use_gpu=bool(use_gpu))
    import tensorflow as tf

    tf.keras.backend.clear_session()

    try:
        look_back_str = str(look_back or DEFAULT_LOOK_BACK)
        window_sizes = []
        for x in look_back_str.split(','):
            x = x.strip()
            if x.isdigit():
                window_sizes.append(int(x))
        if not window_sizes:
            window_sizes = [DEFAULT_LOOK_BACK]
        window_sizes = sorted(list(set(window_sizes)))
    except Exception:
        window_sizes = [DEFAULT_LOOK_BACK]

    stride_val = int(stride or DEFAULT_STRIDE)
    nb_y_val = int(nb_y or DEFAULT_NB_Y)
    first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
    units_val = int(units or DEFAULT_UNITS)
    layers_val = int(layers or DEFAULT_LAYERS)
    lr_val = float(lr or DEFAULT_LEARNING_RATE)
    pred_type = prediction_type or 'return'
    loss_type_val = loss_type or 'mse'
    model_type_val = model_type or 'lstm'
    use_da = use_directional_accuracy if use_directional_accuracy is not None else True

    final_predictions_data = None
    final_model_path = None
    last_meta = None

    for i, look_back_val in enumerate(window_sizes):
        is_last = i == len(window_sizes) - 1
        trainX, trainY, testX, testY, _nb_per_day = prepare_xy_from_store(
            store_json,
            look_back_val,
            stride_val,
            nb_y_val,
            first_minutes_val,
            pred_type,
        )
        if trainX is None or trainX.shape[0] == 0:
            continue

        num_features = trainX.shape[-1]
        scale_factor = 1.0
        if loss_type_val == 'scaled_mse':
            scale_factor = 100.0
            trainY = trainY * scale_factor
            if testY is not None:
                testY = testY * scale_factor

        if model_type_val == 'transformer' and TRANSFORMER_AVAILABLE:
            model = create_transformer_model(
                look_back_val,
                int(num_features),
                nb_y_val,
                int(embed_dim or DEFAULT_EMBED_DIM),
                int(num_heads or DEFAULT_NUM_HEADS),
                int(transformer_layers or DEFAULT_TRANSFORMER_LAYERS),
                int(ff_multiplier or DEFAULT_FF_MULTIPLIER),
                float(dropout or DEFAULT_DROPOUT),
                lr_val,
                use_da,
                pred_type,
            )
        elif model_type_val == 'hybrid' and TRANSFORMER_AVAILABLE:
            model = create_hybrid_lstm_transformer_model(
                look_back_val,
                int(num_features),
                nb_y_val,
                int(hybrid_lstm_units or DEFAULT_UNITS),
                int(hybrid_lstm_layers or DEFAULT_LAYERS),
                int(hybrid_embed_dim or DEFAULT_EMBED_DIM),
                int(hybrid_num_heads or DEFAULT_NUM_HEADS),
                int(hybrid_trans_layers or 1),
                DEFAULT_FF_MULTIPLIER,
                float(hybrid_dropout or DEFAULT_DROPOUT),
                lr_val,
                use_da,
                pred_type,
                fusion_mode or DEFAULT_FUSION_MODE,
            )
        else:
            model = build_lstm_model(
                look_back_val,
                int(num_features),
                nb_y_val,
                units_val,
                layers_val,
                lr_val,
                use_da,
                pred_type,
                loss_type_val,
            )

        num_epochs = int(epochs or DEFAULT_EPOCHS)
        model.fit(
            trainX,
            trainY,
            epochs=num_epochs,
            validation_data=(testX, testY) if (testX is not None and getattr(testX, 'size', 0)) else None,
            verbose=0,
        )

        y_pred = model.predict(testX, verbose=0) if (testX is not None and getattr(testX, 'size', 0)) else None
        if pred_type == 'signal' and y_pred is not None:
            y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)

        if scale_factor != 1.0:
            if y_pred is not None:
                y_pred = y_pred / scale_factor
            if testY is not None:
                testY = testY / scale_factor

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        tmp_path = tmp_file.name
        tmp_file.close()
        try:
            model.save(tmp_path, include_optimizer=False, save_format='h5')
        except Exception:
            tmp_path = None

        final_model_path = tmp_path
        last_meta = {
            'look_back': look_back_val,
            'stride': stride_val,
            'nb_y': nb_y_val,
            'first_minutes': first_minutes_val,
            'prediction_type': pred_type,
            'loss_type': loss_type_val,
            'scale_factor': scale_factor,
        }

        if is_last:
            da_main = None
            if y_pred is not None and testY is not None:
                baseline = 1.0 if pred_type == 'price' else 0.0
                true_dir = np.sign(testY - baseline)
                pred_dir = np.sign(y_pred - baseline)
                da_main = float((true_dir == pred_dir).mean())

            final_predictions_data = {
                'y_pred_test': y_pred.tolist() if y_pred is not None else [],
                'y_true_test': testY.tolist() if testY is not None else [],
                'predictions_flat': y_pred.flatten().tolist() if y_pred is not None else [],
                'look_back': look_back_val,
                'stride': stride_val,
                'nb_y': nb_y_val,
                'first_minutes': first_minutes_val,
                'prediction_type': pred_type,
                'directional_accuracy': da_main,
                'num_epochs': num_epochs,
                'loss_type': loss_type_val,
            }

    if not final_model_path:
        logging.warning("[train_model_impl] Aucun modèle n'a été entraîné (données insuffisantes ?).")

    return {
        'model_path': final_model_path,
        'predictions_data': final_predictions_data,
        'meta': last_meta,
    }


