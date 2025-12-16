# -*- coding: utf-8 -*-
"""
Callbacks d'entraînement de modèles pour le playground.
Ce fichier contient les callbacks lourds (background) pour l'entraînement.

Historique:
- Ce code était initialement dans `web/apps/playground.py`.
- Il a été migré ici pour éviter la duplication et permettre la suppression de `playground.py`.
"""

import logging
import tempfile
import time

import dash
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, State, html

from app import app

from Models.training import (
    build_lstm_model,
    prepare_xy_from_store,
)
from web.services.graphs import build_segments_graph_from_store

# Import des modèles Transformer/Hybride (optionnel)
try:
    from Models.transformer import (
        create_transformer_model,
        create_hybrid_lstm_transformer_model,
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logging.warning("[Playground callbacks] Module transformer non disponible")

from web.apps.model_config import (
    DEFAULT_EPOCHS,
    DEFAULT_FIRST_MINUTES,
    DEFAULT_FF_MULTIPLIER,
    DEFAULT_FUSION_MODE,
    DEFAULT_LOOK_BACK,
    DEFAULT_NB_Y,
    DEFAULT_NUM_HEADS,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LSTM_LAYERS as DEFAULT_LAYERS,
    DEFAULT_LSTM_UNITS as DEFAULT_UNITS,
    DEFAULT_STRIDE,
    DEFAULT_TRANSFORMER_LAYERS,
)

from .. import state as pg_state


@app.callback(
    Output('play_first_minutes', 'value'),
    [
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
    ],
    [State('play_first_minutes', 'value')],
    prevent_initial_call=True,
)
def adjust_first_minutes(look_back, stride, current_first_minutes):
    """
    Ajuste automatiquement 'play_first_minutes' pour respecter :
    first_minutes >= max(look_back) * stride
    """
    try:
        look_back_str = str(look_back or DEFAULT_LOOK_BACK)
        window_sizes = []
        for x in look_back_str.split(','):
            x = x.strip()
            if x.isdigit():
                window_sizes.append(int(x))
        if not window_sizes:
            window_sizes = [DEFAULT_LOOK_BACK]

        max_lb = max(window_sizes)
        st = int(stride or DEFAULT_STRIDE)
        fm = int(current_first_minutes or DEFAULT_FIRST_MINUTES)

        min_required = max_lb * st

        if fm < min_required:
            logging.info(
                "[UI] Auto-adjust: first_minutes (%s) < max_look_back*stride (%s). Updating to %s.",
                fm,
                min_required,
                min_required,
            )
            return min_required

        return dash.no_update
    except Exception:
        return dash.no_update


@app.callback(
    [
        Output('play_train_backtest', 'style'),
        Output('play_stop_training', 'style'),
        Output('play_train_backtest', 'disabled'),
    ],
    [Input('play_reset_buttons', 'data')],
    prevent_initial_call=False,
)
def reset_training_buttons(reset_data):
    """
    Réinitialise l'état des boutons d'entraînement au démarrage de la page.
    """
    _ = reset_data
    return (
        {
            'display': 'block',
            'width': '100%',
            'backgroundColor': '#4CAF50',
            'color': 'white',
            'padding': '12px',
            'fontSize': '14px',
            'fontWeight': 'bold',
            'border': 'none',
            'borderRadius': '8px',
            'cursor': 'pointer',
        },
        {'display': 'none'},
        False,
    )


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
        Output('play_predictions_store', 'data'),
        Output('play_run_backtest', 'disabled'),
        Output('play_model_ready', 'data'),
        Output('play_model_path', 'data'),
    ],
    [Input('play_train_backtest', 'n_clicks')],
    [
        State('play_df_store', 'data'),
        State('play_look_back', 'value'),
        State('play_stride', 'value'),
        State('play_nb_y', 'value'),
        State('play_first_minutes', 'value'),
        State('play_use_directional_accuracy', 'value'),
        State('play_loss_type', 'value'),
        State('play_units', 'value'),
        State('play_layers', 'value'),
        State('play_lr', 'value'),
        State('play_epochs', 'value'),
        State('play_prediction_type', 'value'),
        # Type de modèle
        State('play_model_type', 'value'),
        # Paramètres Transformer
        State('play_embed_dim', 'value'),
        State('play_num_heads', 'value'),
        State('play_transformer_layers', 'value'),
        State('play_ff_multiplier', 'value'),
        State('play_dropout', 'value'),
        # Paramètres Hybride
        State('play_hybrid_lstm_units', 'value'),
        State('play_hybrid_lstm_layers', 'value'),
        State('play_hybrid_embed_dim', 'value'),
        State('play_hybrid_num_heads', 'value'),
        State('play_hybrid_trans_layers', 'value'),
        State('play_fusion_mode', 'value'),
        State('play_hybrid_dropout', 'value'),
        # Choix GPU/CPU
        State('play_use_gpu', 'value'),
    ],
    background=True,
    progress=[
        Output('play_training_progress', 'children'),
        Output('play_training_history', 'figure'),
    ],
    running=[
        (
            Output('play_train_backtest', 'style'),
            {'display': 'none'},
            {
                'display': 'block',
                'width': '100%',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'padding': '12px',
                'fontSize': '14px',
                'fontWeight': 'bold',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
            },
        ),
        (
            Output('play_stop_training', 'style'),
            {
                'display': 'block',
                'width': '100%',
                'backgroundColor': '#EF4444',
                'color': 'white',
                'padding': '12px',
                'fontSize': '14px',
                'fontWeight': 'bold',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
            },
            {'display': 'none'},
        ),
        (Output('play_train_backtest', 'disabled'), True, False),
    ],
    cancel=[Input('play_stop_training', 'n_clicks')],
)
def train_model(
    set_progress,
    n_clicks,
    store_json,
    look_back,
    stride,
    nb_y,
    first_minutes,
    use_directional_accuracy,
    loss_type,
    units,
    layers,
    lr,
    epochs,
    prediction_type,
    model_type,
    embed_dim,
    num_heads,
    transformer_layers,
    ff_multiplier,
    dropout,
    hybrid_lstm_units,
    hybrid_lstm_layers,
    hybrid_embed_dim,
    hybrid_num_heads,
    hybrid_trans_layers,
    fusion_mode,
    hybrid_dropout,
    use_gpu,
):
    """
    Fonction d'entraînement qui s'exécute dans un worker séparé.
    IMPORTANT: on configure CUDA dans le worker avant d'importer TensorFlow.
    """
    from Models.training.tf_worker_setup import setup_cuda_for_worker

    use_gpu_val = bool(use_gpu and 'gpu' in use_gpu)
    setup_cuda_for_worker(use_gpu=use_gpu_val)

    if not use_gpu_val:
        logging.info("[Worker] Mode CPU forcé (checkbox désactivée)")
    else:
        logging.info("[Worker] Mode GPU activé, réinitialisation CUDA...")

    import tensorflow as tf

    try:
        tf.keras.backend.clear_session()
        if use_gpu_val:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError:
                        pass
                logging.info("[Worker] CUDA réinitialisé. GPU disponibles: %s", len(gpus))
            else:
                logging.warning("[Worker] Aucun GPU détecté malgré la demande GPU. Utilisation CPU.")
                use_gpu_val = False
        else:
            logging.info("[Worker] Mode CPU activé")
    except Exception as e:
        logging.warning("[Worker] Erreur réinitialisation CUDA: %s", e)
        use_gpu_val = False

    history_fig = go.Figure()
    history_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        font={'color': '#FFF'},
        title='En attente...',
        height=300,
        uirevision='play_hist',
    )
    empty_seg_fig = go.Figure()
    empty_seg_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        font={'color': '#FFF'},
        title='Segments — en attente',
        height=420,
        uirevision='play_segments',
    )

    if not n_clicks:
        return empty_seg_fig, None, True, False, None

    try:
        if use_gpu_val:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_msg = f"✅ GPU activé ({len(gpus)} GPU disponible(s))" if gpus else "⚠️ GPU demandé mais non disponible, utilisation CPU"
            if not gpus:
                use_gpu_val = False
        else:
            gpu_msg = "💻 CPU activé (checkbox désactivée)"

        set_progress((html.Div(f"ℹ️ Info système: {gpu_msg}"), history_fig))

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

        extra_predictions = []
        final_predictions_data = None
        final_model_path = None
        predictions_flat_main = None
        predictions_train_flat_main = None
        da_main = None

        colors = ['#FF00FF', '#FFFF00', '#00FF00', '#00E0FF', '#FF8C00']

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

        accs, vaccs, losses, vlosses = [], [], [], []

        def _make_hist_fig():
            fig_h = go.Figure()
            if losses:
                fig_h.add_trace(
                    go.Scatter(
                        x=list(range(1, len(losses) + 1)),
                        y=losses,
                        mode='lines+markers',
                        name='Loss train',
                        line={'color': '#2ca02c', 'width': 2},
                        marker={'size': 6},
                        yaxis='y',
                    )
                )
            if vlosses:
                fig_h.add_trace(
                    go.Scatter(
                        x=list(range(1, len(vlosses) + 1)),
                        y=vlosses,
                        mode='lines+markers',
                        name='Loss val',
                        line={'color': '#d62728', 'width': 2},
                        marker={'size': 6},
                        yaxis='y',
                    )
                )

            if accs:
                accs_pct = [a * 100 for a in accs]
                fig_h.add_trace(
                    go.Scatter(
                        x=list(range(1, len(accs_pct) + 1)),
                        y=accs_pct,
                        mode='lines+markers',
                        name='DA train %',
                        line={'color': '#1f77b4', 'width': 2, 'dash': 'dot'},
                        marker={'size': 6},
                        yaxis='y2',
                    )
                )
            if vaccs:
                vaccs_pct = [a * 100 for a in vaccs]
                fig_h.add_trace(
                    go.Scatter(
                        x=list(range(1, len(vaccs_pct) + 1)),
                        y=vaccs_pct,
                        mode='lines+markers',
                        name='DA val %',
                        line={'color': '#ff7f0e', 'width': 2, 'dash': 'dot'},
                        marker={'size': 6},
                        yaxis='y2',
                    )
                )

            y_cfg = {'title': 'Loss', 'side': 'left', 'type': 'log'}
            loss_info = ''
            if losses or vlosses:
                all_loss = [l for l in (losses + vlosses) if l is not None and l > 0]
                if all_loss:
                    current_loss = float(all_loss[-1])
                    loss_info = f" (actuel: {current_loss:.2e})" if current_loss < 0.001 else f" (actuel: {current_loss:.6f})"

            fig_h.update_layout(
                template='plotly_dark',
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font={'color': '#FFFFFF'},
                title=f'📊 Loss{loss_info} & DA',
                height=300,
                uirevision='play_hist',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    font={'size': 10},
                ),
                margin=dict(t=60, b=40, l=60, r=60),
                yaxis=y_cfg,
                yaxis2={
                    'title': 'DA %',
                    'overlaying': 'y',
                    'side': 'right',
                    'range': [0, 100],
                    'ticksuffix': '%',
                },
            )
            return fig_h

        for i, look_back_val in enumerate(window_sizes):
            is_last = i == len(window_sizes) - 1
            msg_prefix = f"[Modèle {i+1}/{len(window_sizes)} Win={look_back_val}] "
            set_progress((html.Div(f'{msg_prefix}Préparation des données...'), history_fig))

            trainX, trainY, testX, testY, _nb_per_day = prepare_xy_from_store(
                store_json,
                look_back_val,
                stride_val,
                nb_y_val,
                first_minutes_val,
                pred_type,
            )

            if trainX is None or trainX.shape[0] == 0:
                logging.warning("%sPas de données, skipping.", msg_prefix)
                continue

            num_features = trainX.shape[-1]
            scale_factor = 1.0
            if loss_type_val == 'scaled_mse':
                scale_factor = 100.0
                trainY = trainY * scale_factor
                if testY is not None:
                    testY = testY * scale_factor

            set_progress((html.Div(f'{msg_prefix}Construction du modèle ({model_type_val})...'), history_fig))

            if model_type_val == 'transformer' and TRANSFORMER_AVAILABLE:
                embed_dim_val = int(embed_dim or DEFAULT_EMBED_DIM)
                num_heads_val = int(num_heads or DEFAULT_NUM_HEADS)
                trans_layers_val = int(transformer_layers or DEFAULT_TRANSFORMER_LAYERS)
                ff_mult_val = int(ff_multiplier or DEFAULT_FF_MULTIPLIER)
                dropout_val = float(dropout or DEFAULT_DROPOUT)
                model = create_transformer_model(
                    look_back_val,
                    int(num_features),
                    nb_y_val,
                    embed_dim_val,
                    num_heads_val,
                    trans_layers_val,
                    ff_mult_val,
                    dropout_val,
                    lr_val,
                    use_da,
                    pred_type,
                )
            elif model_type_val == 'hybrid' and TRANSFORMER_AVAILABLE:
                h_lstm_units = int(hybrid_lstm_units or DEFAULT_UNITS)
                h_lstm_layers = int(hybrid_lstm_layers or DEFAULT_LAYERS)
                h_embed_dim = int(hybrid_embed_dim or DEFAULT_EMBED_DIM)
                h_num_heads = int(hybrid_num_heads or DEFAULT_NUM_HEADS)
                h_trans_layers = int(hybrid_trans_layers or 1)
                h_fusion = fusion_mode or DEFAULT_FUSION_MODE
                h_dropout = float(hybrid_dropout or DEFAULT_DROPOUT)
                model = create_hybrid_lstm_transformer_model(
                    look_back_val,
                    int(num_features),
                    nb_y_val,
                    h_lstm_units,
                    h_lstm_layers,
                    h_embed_dim,
                    h_num_heads,
                    h_trans_layers,
                    DEFAULT_FF_MULTIPLIER,
                    h_dropout,
                    lr_val,
                    use_da,
                    pred_type,
                    h_fusion,
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
            set_progress((html.Div(f'{msg_prefix}Entraînement ({num_epochs} epochs)...'), history_fig))

            class FullProgCB(tf.keras.callbacks.Callback):
                def __init__(self, total_epochs, metric_losses, metric_vlosses, metric_accs, metric_vaccs):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.losses = metric_losses
                    self.vlosses = metric_vlosses
                    self.accs = metric_accs
                    self.vaccs = metric_vaccs
                    self.last_update = time.time()

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    l = logs.get('loss')
                    vl = logs.get('val_loss')
                    a = logs.get('accuracy') or logs.get('directional_accuracy')
                    va = logs.get('val_accuracy') or logs.get('val_directional_accuracy')

                    if l is not None:
                        self.losses.append(float(l))
                    if vl is not None:
                        self.vlosses.append(float(vl))
                    if a is not None:
                        self.accs.append(float(a))
                    if va is not None:
                        self.vaccs.append(float(va))

                    if (time.time() - self.last_update > 0.5) or (epoch == self.total_epochs - 1):
                        self.last_update = time.time()
                        new_fig = _make_hist_fig()
                        loss_txt = f"{l:.4f}" if l else "?"
                        set_progress((html.Div(f"{msg_prefix}Epoch {epoch+1}/{self.total_epochs} - Loss={loss_txt}"), new_fig))

            model.fit(
                trainX,
                trainY,
                epochs=num_epochs,
                validation_data=(testX, testY) if (testX is not None and getattr(testX, 'size', 0)) else None,
                verbose=0,
                callbacks=[FullProgCB(num_epochs, losses, vlosses, accs, vaccs)],
            )

            y_pred = model.predict(testX, verbose=0) if (testX is not None and getattr(testX, 'size', 0)) else None
            y_pred_train = model.predict(trainX, verbose=0) if (trainX is not None and getattr(trainX, 'size', 0)) else None

            if pred_type == 'signal':
                if y_pred is not None:
                    y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
                if y_pred_train is not None:
                    y_pred_train = np.argmax(y_pred_train, axis=1).reshape(-1, 1)

            if scale_factor != 1.0:
                if y_pred is not None:
                    y_pred = y_pred / scale_factor
                if y_pred_train is not None:
                    y_pred_train = y_pred_train / scale_factor
                if testY is not None:
                    testY = testY / scale_factor

            preds_flat = y_pred.flatten().tolist() if y_pred is not None else []
            preds_train_flat = y_pred_train.flatten().tolist() if y_pred_train is not None else []

            pg_state.play_last_model = model
            pg_state.play_last_model_meta = {
                'look_back': look_back_val,
                'stride': stride_val,
                'nb_y': nb_y_val,
                'first_minutes': first_minutes_val,
                'prediction_type': pred_type,
                'loss_type': loss_type_val,
                'scale_factor': scale_factor,
            }
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            tmp_path = tmp_file.name
            tmp_file.close()
            try:
                model.save(tmp_path, include_optimizer=False, save_format='h5')
            except Exception:
                tmp_path = None
            final_model_path = tmp_path

            if is_last:
                if y_pred is not None and testY is not None:
                    baseline = 1.0 if pred_type == 'price' else 0.0
                    true_dir = np.sign(testY - baseline)
                    pred_dir = np.sign(y_pred - baseline)
                    da_main = float((true_dir == pred_dir).mean())

                predictions_flat_main = preds_flat
                predictions_train_flat_main = preds_train_flat

                final_predictions_data = {
                    'y_pred_test': y_pred.tolist() if y_pred is not None else [],
                    'y_true_test': testY.tolist() if testY is not None else [],
                    'predictions_flat': predictions_flat_main,
                    'predictions_train_flat': predictions_train_flat_main,
                    'look_back': look_back_val,
                    'stride': stride_val,
                    'nb_y': nb_y_val,
                    'first_minutes': first_minutes_val,
                    'prediction_type': pred_type,
                    'directional_accuracy': da_main,
                    'num_epochs': num_epochs,
                    'loss_type': loss_type_val,
                }
            else:
                extra_predictions.append(
                    {
                        'test': preds_flat,
                        'name': f'Win={look_back_val}',
                        'color': colors[i % len(colors)],
                    }
                )

        loss_label = {'mse': 'MSE', 'scaled_mse': 'Scaled MSE', 'mae': 'MAE'}.get(loss_type_val, 'MSE')
        title = f'📊 Prédictions ({loss_label})'
        if da_main:
            title += f' — DA={da_main*100:.1f}%'

        seg_fig = build_segments_graph_from_store(
            store_json,
            window_sizes[-1],
            stride_val,
            first_minutes_val,
            predictions=predictions_flat_main,
            nb_y=nb_y_val,
            predictions_train=predictions_train_flat_main,
            prediction_type=pred_type,
            extra_predictions=extra_predictions,
        )
        seg_fig.update_layout(title=title)

        logging.info("[Training] Tout terminé.")
        return seg_fig, final_predictions_data, False, True, final_model_path

    except Exception as e:
        logging.error("[Training] Erreur: %s", e)
        err_msg = str(e)
        if "CUDA_ERROR_NOT_INITIALIZED" in err_msg:
            err_msg = "Erreur CUDA (GPU). Veuillez redémarrer l'application pour appliquer le mode 'spawn'."

        empty_seg_fig = go.Figure()
        empty_seg_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000',
            plot_bgcolor='#000',
            font={'color': '#FFF'},
            title=f'❌ Erreur: {err_msg}',
            height=420,
            uirevision='play_segments',
        )
        return empty_seg_fig, None, True, False, None


