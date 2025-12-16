# -*- coding: utf-8 -*-
"""
Callbacks liés aux modèles sauvegardés (BDD) et à certains éléments UI.

Migré depuis `web/apps/playground.py` pour éviter la duplication.
"""

import json
import logging

import plotly.graph_objs as go
from dash import Input, Output, State

from app import app, shM
from web.apps.model_config import (
    DEFAULT_FIRST_MINUTES,
    DEFAULT_LOOK_BACK,
    DEFAULT_NB_Y,
    DEFAULT_STRIDE,
    MODEL_TYPES,
)
from web.services.graphs import build_segments_graph_from_store


@app.callback(
    [
        Output('play_saved_model', 'options'),
        Output('play_saved_model', 'value'),
    ],
    [
        Input('play_symbol', 'value'),
        Input('play_saved_model_type_filter', 'value'),
    ],
)
def populate_saved_models(symbol, model_type_filter):
    """Remplit la liste des modèles sauvegardés en fonction du symbole et du type."""
    options = []
    try:
        rows = []
        try:
            if model_type_filter and model_type_filter != 'all':
                rows = shM.list_models_by_type(model_type_filter)
            elif symbol:
                rows = shM.list_models_for_symbol(symbol)
            else:
                rows = shM.list_models_by_type(None)
        except Exception as db_err:
            # Fallback si la colonne model_type n'existe pas dans la DB
            logging.warning("Error while listing models by type: %s", db_err)
            try:
                rows = shM.list_models_for_symbol(symbol) if symbol else []
            except Exception:
                rows = []

        for row in rows:
            # Format typique: (id, date, trainScore, testScore, model_type, symbols)
            mid = row[0]
            date_val = row[1] if len(row) > 1 else None
            train_s = row[2] if len(row) > 2 else None
            test_s = row[3] if len(row) > 3 else None
            m_type = row[4] if len(row) > 4 else 'lstm'
            symbols_json = row[5] if len(row) > 5 else None

            if symbol:
                symbols_list = []
                if symbols_json:
                    try:
                        symbols_list = json.loads(symbols_json) if isinstance(symbols_json, str) else symbols_json
                    except Exception:
                        symbols_list = []
                if symbols_list and symbol not in symbols_list:
                    continue

            type_emoji = MODEL_TYPES.get(m_type, {}).get('icon', '❓')

            train_str = f"{train_s:.4f}" if train_s is not None else '-'
            test_str = f"{test_s:.4f}" if test_s is not None else '-'
            label = f"{type_emoji} {mid} — {str(date_val)[:10]} — train={train_str} test={test_str}"
            options.append(
                {
                    'label': label,
                    'value': mid,
                }
            )

        return options, (options[0]['value'] if options else None)
    except Exception:
        return [], None


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
    ],
    [
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
        Input('play_first_minutes', 'value'),
        Input('play_nb_y', 'value'),
        Input('play_prediction_type', 'value'),
    ],
    [
        State('play_df_store', 'data'),
    ],
    prevent_initial_call=True,
)
def update_segments_graph(look_back, stride, first_minutes, nb_y, prediction_type, store_json):
    """Met à jour le graphe des segments quand les paramètres changent."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        font={
            'color': '#FFF',
        },
        title="Générer une courbe d'abord",
        height=420,
        uirevision='play_segments',
    )

    if not store_json:
        return (empty_fig,)

    try:
        look_back_val = int(look_back or DEFAULT_LOOK_BACK)
        stride_val = int(stride or DEFAULT_STRIDE)
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
        nb_y_val = int(nb_y or DEFAULT_NB_Y)
        pred_type = prediction_type or 'return'
        fig = build_segments_graph_from_store(
            store_json,
            look_back_val,
            stride_val,
            first_minutes_val,
            None,
            nb_y_val,
            None,
            pred_type,
        )
        return (fig,)
    except Exception:
        return (empty_fig,)


@app.callback(
    [
        Output('play_nb_y', 'max'),
        Output('play_nb_y', 'value'),
        Output('play_nb_y', 'marks'),
    ],
    [
        Input('play_first_minutes', 'value'),
        Input('play_open_time', 'value'),
        Input('play_close_time', 'value'),
    ],
    [
        State('play_nb_y', 'value'),
    ],
)
def update_nb_y_slider(first_minutes, open_time, close_time, current_nb_y):
    """
    Ajuste le slider nb_y selon la durée de journée et first_minutes.
    (L'affichage texte est géré séparément par `update_nb_y_display`).
    """
    try:
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
    except Exception:
        first_minutes_val = DEFAULT_FIRST_MINUTES

    def parse_minutes(hhmm, fallback):
        try:
            parts = str(hhmm or '').split(':')
            h = int(parts[0])
            m = int(parts[1])
            return h * 60 + m
        except Exception:
            return fallback

    open_min = parse_minutes(open_time, 9 * 60 + 30)
    close_min = parse_minutes(close_time, 16 * 60)
    day_len = max(0, close_min - open_min)
    remainder = max(0, day_len - max(0, first_minutes_val))

    max_nb_y = max(1, max(0, remainder - 1))

    try:
        cur_val = int(current_nb_y or 5)
    except Exception:
        cur_val = 5

    new_val = min(max_nb_y, max(1, cur_val))

    marks = {
        1: '1',
        max_nb_y: str(max_nb_y),
    }

    logging.info(
        "[UI] Ajustement slider nb_y — day_len=%s remainder=%s max=%s value=%s",
        day_len,
        remainder,
        max_nb_y,
        new_val,
    )
    return max_nb_y, new_val, marks


@app.callback(
    Output('play_nb_y_value', 'children'),
    Input('play_nb_y', 'value'),
    prevent_initial_call=False,
)
def update_nb_y_display(nb_y_value):
    """Affiche la valeur actuelle du slider nb_y."""
    try:
        val = int(nb_y_value or 5)
    except Exception:
        val = 5
    return f"Valeur actuelle: {val}"


