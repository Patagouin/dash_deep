from dash.dependencies import Input, Output, State
from app import app, shM
import logging
import json
import os


@app.callback(
    Output('prediction_dropdown', 'options'),
    Input('sector_dropdown', 'value')
)
def update_shares_based_on_sector(selected_sector):
    if selected_sector == 'Non défini':
        filtered_shares = shM.dfShares[shM.dfShares['sector'].isna()]
    elif selected_sector:
        filtered_shares = shM.dfShares[shM.dfShares['sector'] == selected_sector]
    else:
        filtered_shares = shM.dfShares

    sorted_shares = filtered_shares.sort_values(by='symbol')
    return [{'label': 'All', 'value': 'All'}] + [
        {'label': f'{stock.symbol}', 'value': stock.symbol}
        for stock in sorted_shares.itertuples()
    ]


@app.callback(
    Output('training_days_slider', 'max'),
    Output('training_days_slider', 'marks'),
    Input('train_share_list', 'value')
)
def update_days_slider(selected_symbols):
    if not selected_symbols:
        return 365, {
            7: '7j',
            30: '1m',
            90: '3m',
            180: '6m',
            365: '1a'
        }

    max_days = 0
    for symbol in selected_symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if not dfShares.empty:
            dfData = shM.getListDfDataFromDfShares(dfShares)[0]
            if not dfData.empty:
                days_available = (dfData.index.max() - dfData.index.min()).days
                max_days = max(max_days, days_available)

    max_days = min(((max_days + 29) // 30) * 30, 365)

    marks = {
        7: '7j',
        30: '1m',
    }

    if max_days >= 90:
        marks[90] = '3m'
    if max_days >= 180:
        marks[180] = '6m'
    if max_days >= 365:
        marks[365] = '1a'

    return max_days, marks


@app.callback(
    Output('stats_share_list', 'options'),
    Input('prediction_dropdown', 'value')
)
def update_stats_share_list(selected_symbols):
    try:
        if not selected_symbols:
            return []
        return [
            {'label': symbol, 'value': symbol}
            for symbol in selected_symbols if symbol != 'All'
        ]
    except Exception as e:
        logging.error(f"Error updating stats share list: {e}")
        return []


@app.callback(
    Output('stats_share_list', 'value'),
    Input('prediction_dropdown', 'value')
)
def sync_stats_share_selection(selected_symbols):
    if not selected_symbols:
        return []
    return [s for s in selected_symbols if s != 'All']
# Options d'entraînement synchronisées avec le secteur
@app.callback(
    Output('train_share_list', 'options'),
    Input('sector_dropdown', 'value')
)
def update_train_list_options(selected_sector):
    if selected_sector == 'Non défini':
        filtered_shares = shM.dfShares[shM.dfShares['sector'].isna()]
    elif selected_sector:
        filtered_shares = shM.dfShares[shM.dfShares['sector'] == selected_sector]
    else:
        filtered_shares = shM.dfShares

    sorted_shares = filtered_shares.sort_values(by='symbol')
    return [
        {'label': f'{stock.symbol}', 'value': stock.symbol}
        for stock in sorted_shares.itertuples()
    ]

# Helpers pour presets
def _conf_path():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))

def _load_conf():
    try:
        with open(_conf_path(), 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_conf(conf):
    with open(_conf_path(), 'w') as f:
        json.dump(conf, f, indent=2)

def _add_value(saved_list, value):
    vals = saved_list or []
    if value is None:
        return vals
    if value not in vals:
        vals.append(value)
    return vals

# -- Model type options management (avec icônes) --
@app.callback(Output('model_type_saved', 'options'), Output('model_type_saved', 'value'), Input('add_model_type', 'n_clicks'), State('model_type', 'value'), State('model_type_saved', 'value'), prevent_initial_call=True)
def add_model_type(n, value, current_selected):
    # Mapping avec icônes pour les types de modèles
    mapping = {
        'lstm': '🔄 LSTM',
        'gru': '🔃 GRU',
        'transformer': '🎯 Transformer',
        'hybrid': '🔀 Hybride'
    }
    if not n:
        conf = _load_conf()
        vals = conf.get('model_type_options', [])
        return [{'label': mapping.get(v, v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('model_type_options', []), value)
    conf['model_type_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': mapping.get(v, v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

# -- Advanced Transformer params
@app.callback(Output('transformer_num_heads_saved', 'options'), Output('transformer_num_heads_saved', 'value'), Input('add_transformer_num_heads', 'n_clicks'), State('transformer_num_heads', 'value'), State('transformer_num_heads_saved', 'value'), prevent_initial_call=True)
def add_transformer_num_heads(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('transformer_num_heads_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('transformer_num_heads_options', []), value)
    conf['transformer_num_heads_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

@app.callback(Output('transformer_dropout_saved', 'options'), Output('transformer_dropout_saved', 'value'), Input('add_transformer_dropout', 'n_clicks'), State('transformer_dropout', 'value'), State('transformer_dropout_saved', 'value'), prevent_initial_call=True)
def add_transformer_dropout(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('transformer_dropout_options', [])
        return [{'label': f"{float(v):.1f}", 'value': float(v)} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('transformer_dropout_options', []), value)
    conf['transformer_dropout_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': f"{float(v):.1f}", 'value': float(v)} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

@app.callback(Output('transformer_ff_multiplier_saved', 'options'), Output('transformer_ff_multiplier_saved', 'value'), Input('add_transformer_ff_multiplier', 'n_clicks'), State('transformer_ff_multiplier', 'value'), State('transformer_ff_multiplier_saved', 'value'), prevent_initial_call=True)
def add_transformer_ff_multiplier(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('transformer_ff_multiplier_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('transformer_ff_multiplier_options', []), value)
    conf['transformer_ff_multiplier_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected
@app.callback(Output('training_days_saved', 'options'), Output('training_days_saved', 'value'), Input('add_training_days', 'n_clicks'), State('training_days_slider', 'value'), State('training_days_saved', 'value'), prevent_initial_call=True)
def add_training_days(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('training_days_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('training_days_options', []), value)
    conf['training_days_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

@app.callback(Output('train_test_ratio_saved', 'options'), Output('train_test_ratio_saved', 'value'), Input('add_train_ratio', 'n_clicks'), State('train_test_ratio_slider', 'value'), State('train_test_ratio_saved', 'value'), prevent_initial_call=True)
def add_train_ratio(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('train_test_ratio_options', [])
        return [{'label': f"{v}%", 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('train_test_ratio_options', []), value)
    conf['train_test_ratio_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': f"{v}%", 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('look_back_saved', 'options'), Output('look_back_saved', 'value'), Input('add_look_back', 'n_clicks'), State('look_back_x', 'value'), State('look_back_saved', 'value'), prevent_initial_call=True)
def add_look_back(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('look_back_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('look_back_options', []), value)
    conf['look_back_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('stride_saved', 'options'), Output('stride_saved', 'value'), Input('add_stride', 'n_clicks'), State('stride_x', 'value'), State('stride_saved', 'value'), prevent_initial_call=True)
def add_stride(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('stride_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('stride_options', []), value)
    conf['stride_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('nb_y_saved', 'options'), Output('nb_y_saved', 'value'), Input('add_nb_y', 'n_clicks'), State('nb_y', 'value'), State('nb_y_saved', 'value'), prevent_initial_call=True)
def add_nb_y(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('nb_y_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('nb_y_options', []), value)
    conf['nb_y_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('nb_units_saved', 'options'), Output('nb_units_saved', 'value'), Input('add_nb_units', 'n_clicks'), State('nb_units', 'value'), State('nb_units_saved', 'value'), prevent_initial_call=True)
def add_nb_units(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('nb_units_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('nb_units_options', []), value)
    conf['nb_units_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('layers_saved', 'options'), Output('layers_saved', 'value'), Input('add_layers', 'n_clicks'), State('layers', 'value'), State('layers_saved', 'value'), prevent_initial_call=True)
def add_layers(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('layers_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('layers_options', []), value)
    conf['layers_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('learning_rate_saved', 'options'), Output('learning_rate_saved', 'value'), Input('add_learning_rate', 'n_clicks'), State('learning_rate', 'value'), State('learning_rate_saved', 'value'), prevent_initial_call=True)
def add_learning_rate(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('learning_rate_options', [])
        return [{'label': f"{v:.1e}", 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('learning_rate_options', []), value)
    conf['learning_rate_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': f"{v:.1e}", 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('loss_saved', 'options'), Output('loss_saved', 'value'), Input('add_loss', 'n_clicks'), State('loss_function', 'value'), State('loss_saved', 'value'), prevent_initial_call=True)
def add_loss(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('loss_options', [])
        mapping = {'mse': 'MSE', 'mae': 'MAE', 'huber_loss': 'Huber'}
        return [{'label': mapping.get(v, v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('loss_options', []), value)
    conf['loss_options'] = new_vals
    _save_conf(conf)
    mapping = {'mse': 'MSE', 'mae': 'MAE', 'huber_loss': 'Huber'}
    opts = [{'label': mapping.get(v, v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(Output('epochs_saved', 'options'), Output('epochs_saved', 'value'), Input('add_epochs', 'n_clicks'), State('epochs', 'value'), State('epochs_saved', 'value'), prevent_initial_call=True)
def add_epochs(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('epochs_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('epochs_options', []), value)
    conf['epochs_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

# -- Trading strategy params --
@app.callback(Output('trade_volume_saved', 'options'), Output('trade_volume_saved', 'value'), Input('add_trade_volume', 'n_clicks'), State('trade_volume', 'value'), State('trade_volume_saved', 'value'), prevent_initial_call=True)
def add_trade_volume(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('trade_volume_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('trade_volume_options', []), value)
    conf['trade_volume_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

@app.callback(Output('k_trades_saved', 'options'), Output('k_trades_saved', 'value'), Input('add_k_trades', 'n_clicks'), State('k_trades', 'value'), State('k_trades_saved', 'value'), prevent_initial_call=True)
def add_k_trades(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('k_trades_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('k_trades_options', []), value)
    conf['k_trades_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected

# -- Loss category filtering --
@app.callback(Output('loss_function', 'options'), Input('loss_category', 'value'), prevent_initial_call=False)
def set_loss_options_by_category(category):
    try:
        if category == 'regression':
            return [
                {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
                {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
                {'label': 'Huber Loss', 'value': 'huber_loss'}
            ]
        # Fallback: même liste
        return [
            {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
            {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
            {'label': 'Huber Loss', 'value': 'huber_loss'}
        ]
    except Exception as e:
        logging.error(f"Error setting loss options by category: {e}")
        return [
            {'label': 'Mean Squared Error (MSE)', 'value': 'mse'},
            {'label': 'Mean Absolute Error (MAE)', 'value': 'mae'},
            {'label': 'Huber Loss', 'value': 'huber_loss'}
        ]


# ==============================================================================
# Paramètres Hybride LSTM+Transformer
# ==============================================================================

@app.callback(
    Output('hybrid_lstm_units_saved', 'options'),
    Output('hybrid_lstm_units_saved', 'value'),
    Input('add_hybrid_lstm_units', 'n_clicks'),
    State('hybrid_lstm_units', 'value'),
    State('hybrid_lstm_units_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_lstm_units(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_lstm_units_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_lstm_units_options', []), value)
    conf['hybrid_lstm_units_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(
    Output('hybrid_lstm_layers_saved', 'options'),
    Output('hybrid_lstm_layers_saved', 'value'),
    Input('add_hybrid_lstm_layers', 'n_clicks'),
    State('hybrid_lstm_layers', 'value'),
    State('hybrid_lstm_layers_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_lstm_layers(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_lstm_layers_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_lstm_layers_options', []), value)
    conf['hybrid_lstm_layers_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(
    Output('hybrid_embed_dim_saved', 'options'),
    Output('hybrid_embed_dim_saved', 'value'),
    Input('add_hybrid_embed_dim', 'n_clicks'),
    State('hybrid_embed_dim', 'value'),
    State('hybrid_embed_dim_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_embed_dim(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_embed_dim_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_embed_dim_options', []), value)
    conf['hybrid_embed_dim_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(
    Output('hybrid_num_heads_saved', 'options'),
    Output('hybrid_num_heads_saved', 'value'),
    Input('add_hybrid_num_heads', 'n_clicks'),
    State('hybrid_num_heads', 'value'),
    State('hybrid_num_heads_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_num_heads(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_num_heads_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_num_heads_options', []), value)
    conf['hybrid_num_heads_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(
    Output('hybrid_trans_layers_saved', 'options'),
    Output('hybrid_trans_layers_saved', 'value'),
    Input('add_hybrid_trans_layers', 'n_clicks'),
    State('hybrid_trans_layers', 'value'),
    State('hybrid_trans_layers_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_trans_layers(n, value, current_selected):
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_trans_layers_options', [])
        return [{'label': str(v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_trans_layers_options', []), value)
    conf['hybrid_trans_layers_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': str(v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


@app.callback(
    Output('hybrid_fusion_mode_saved', 'options'),
    Output('hybrid_fusion_mode_saved', 'value'),
    Input('add_hybrid_fusion_mode', 'n_clicks'),
    State('hybrid_fusion_mode', 'value'),
    State('hybrid_fusion_mode_saved', 'value'),
    prevent_initial_call=True
)
def add_hybrid_fusion_mode(n, value, current_selected):
    fusion_labels = {'concat': 'Concat', 'add': 'Add', 'attention': 'Attention'}
    if not n:
        conf = _load_conf()
        vals = conf.get('hybrid_fusion_mode_options', [])
        return [{'label': fusion_labels.get(v, v), 'value': v} for v in vals], vals
    conf = _load_conf()
    new_vals = _add_value(conf.get('hybrid_fusion_mode_options', []), value)
    conf['hybrid_fusion_mode_options'] = new_vals
    _save_conf(conf)
    opts = [{'label': fusion_labels.get(v, v), 'value': v} for v in new_vals]
    selected = list(current_selected or [])
    if value is not None and value not in selected:
        selected.append(value)
    return opts, selected


# ==============================================================================
# Callback pour afficher/masquer les sections de paramètres selon le type de modèle
# ==============================================================================

@app.callback(
    Output('hybrid_params_details', 'style'),
    Input('model_type', 'value'),
    prevent_initial_call=False
)
def toggle_hybrid_params_visibility(model_type):
    """Affiche ou masque la section des paramètres Hybride selon le type de modèle sélectionné."""
    base_style = {'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '8px'}
    if model_type == 'hybrid':
        return base_style
    else:
        return {**base_style, 'display': 'none'}

