# -*- coding: utf-8 -*-
"""
Callbacks de prédiction sur la courbe actuelle pour le playground.
Migré depuis `web/apps/playground.py`.
"""

import logging

import dash
import plotly.graph_objs as go
from dash import Input, Output, State, html

from app import app
from Models.training import prepare_xy_for_inference
from web.apps.model_config import (
    DEFAULT_FIRST_MINUTES,
    DEFAULT_LOOK_BACK,
    DEFAULT_NB_Y,
    DEFAULT_STRIDE,
)
from web.services.graphs import (
    build_segments_graph_from_store,
)

from .. import state as pg_state

try:
    from Models.transformer import get_custom_objects
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
        Output('play_predictions_store', 'data'),
        Output('play_run_backtest', 'disabled'),
        Output('play_gen_summary', 'children'),
    ],
    [
        Input('play_test_generalization', 'n_clicks'),
    ],
    [
        State('play_df_store', 'data'),
        State('play_model_path', 'data'),
        State('play_model_ready', 'data'),
    ],
    prevent_initial_call=True,
)
def predict_on_current_curve(n_clicks, store_json, model_path, model_ready):
    """
    Applique le dernier modèle entraîné sur la courbe synthétique actuelle via model.predict.
    Objectif: produire des prédictions (sans "génération" d'une nouvelle courbe) et alimenter le backtest.
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={'color': '#FFFFFF'},
        title='Série synthétique — en attente de prédiction',
        height=420,
        uirevision='play_segments',
    )

    if not n_clicks:
        return empty_fig, dash.no_update, True, html.Div("Cliquez sur le bouton pour lancer une prédiction.", style={'color': '#CCCCCC'})

    if not model_ready:
        return empty_fig, dash.no_update, True, html.Div("Aucun modèle en mémoire. Entraînez d'abord un modèle.", style={'color': '#F59E0B'})

    if pg_state.play_last_model is None:
        if not model_path:
            return empty_fig, dash.no_update, True, html.Div("Aucun modèle en mémoire. Entraînez d'abord un modèle.", style={'color': '#F59E0B'})
        try:
            import tensorflow as tf

            custom_objects = get_custom_objects() if TRANSFORMER_AVAILABLE else None
            pg_state.play_last_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            pg_state.play_last_model_path = model_path
            logging.info("[Predict] Modèle rechargé depuis %s", model_path)
        except Exception as load_err:
            logging.error("[Predict] Impossible de recharger le modèle: %s", load_err)
            return empty_fig, dash.no_update, True, html.Div("Impossible de recharger le modèle. Ré-entraînez-le.", style={'color': '#EF4444'})

    if not store_json:
        return empty_fig, dash.no_update, True, html.Div("Aucune courbe disponible. Générez d'abord une courbe.", style={'color': '#F59E0B'})

    try:
        look_back = int(pg_state.play_last_model_meta.get('look_back', DEFAULT_LOOK_BACK))
        stride = int(pg_state.play_last_model_meta.get('stride', DEFAULT_STRIDE))
        nb_y = int(pg_state.play_last_model_meta.get('nb_y', DEFAULT_NB_Y))
        first_minutes = int(pg_state.play_last_model_meta.get('first_minutes', DEFAULT_FIRST_MINUTES))
        prediction_type = pg_state.play_last_model_meta.get('prediction_type', 'return')
        loss_type = pg_state.play_last_model_meta.get('loss_type', 'mse')
        scale_factor = float(pg_state.play_last_model_meta.get('scale_factor', 1.0))

        X, Y, df, obs_window, sample_days = prepare_xy_for_inference(
            store_json,
            look_back,
            stride,
            nb_y,
            first_minutes,
            prediction_type,
        )

        if X is None or X.shape[0] == 0:
            return empty_fig, dash.no_update, True, html.Div("Données insuffisantes pour construire des fenêtres d'entrée.", style={'color': '#F59E0B'})

        y_pred = pg_state.play_last_model.predict(X, verbose=0)

        if loss_type == 'scaled_mse' and scale_factor:
            y_pred = y_pred / scale_factor
            if Y is not None:
                Y = Y / scale_factor

        # Construire figure (segments + reconstruction des points prédits)
        fig_base = build_segments_graph_from_store(
            store_json,
            look_back,
            stride,
            first_minutes,
            predictions=y_pred.flatten().tolist(),
            nb_y=nb_y,
            prediction_type=prediction_type,
        )

        fig_base.update_layout(
            legend=dict(
                font=dict(color='#FFFFFF'),
                bgcolor='rgba(0,0,0,0.35)',
                bordercolor='#444',
            ),
            title="Série synthétique & Segments (avec prédiction)",
        )

        # Produire un predictions_store compatible backtest: on garde uniquement la partie "test"
        # (même logique que build_segments_graph_from_store: split 80/20 par jours)
        days = df.index.normalize().unique()
        split_idx = int(len(days) * 0.8)
        test_days = set(days[split_idx:]) if split_idx >= 0 else set()

        test_mask = [d in test_days for d in sample_days]
        y_pred_test = y_pred[test_mask] if hasattr(y_pred, '__getitem__') else y_pred
        y_true_test = Y[test_mask] if (Y is not None and hasattr(Y, '__getitem__')) else None

        predictions_data = {
            'y_pred_test': y_pred_test.tolist() if y_pred_test is not None else [],
            'y_true_test': y_true_test.tolist() if y_true_test is not None else [],
            'predictions_flat': (y_pred_test.flatten().tolist() if y_pred_test is not None else []),
            'look_back': look_back,
            'stride': stride,
            'nb_y': nb_y,
            'first_minutes': first_minutes,
            'prediction_type': prediction_type,
            'loss_type': loss_type,
        }

        summary = html.Div(
            [
                html.Div(f"Prédictions générées : {int(y_pred.shape[0])} jours (échantillons)."),
                html.Div("Backtest activé (basé sur la partie test 80/20)."),
            ],
            style={'color': '#FFFFFF'},
        )

        return fig_base, predictions_data, False, summary
    except Exception as e:
        logging.error("[Predict] Erreur: %s", e)
        return empty_fig, dash.no_update, True, html.Div(f"Erreur durant la prédiction: {e}", style={'color': '#EF4444'})


@app.callback(
    Output('play_test_generalization', 'disabled'),
    [
        Input('play_model_ready', 'data'),
        Input('play_df_store', 'data'),
    ],
)
def toggle_predict_button_disabled(model_ready, store_json):
    """Grise le bouton tant qu'on n'a pas à la fois un modèle et une courbe."""
    return (not bool(model_ready)) or (not bool(store_json))

