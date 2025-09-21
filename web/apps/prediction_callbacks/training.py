from dash.dependencies import Input, Output, State
from dash import html, dash_table
import dash
import plotly.graph_objs as go
from app import app, shM, socketio
from Models import prediction_utils as pred_ut
from tensorflow.keras.callbacks import Callback
import traceback
import logging
import json
import os


@app.callback(
    Output('preset_dropdown', 'options'),
    [Input('preset_dropdown', 'id'), Input('config_bootstrap', 'n_intervals')]
)
def populate_preset_dropdown(_, __):
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conf = {}
    
    presets = conf.get('presets', {})
    preset_options = [{'label': name, 'value': name} for name in presets.keys()]
    try:
        socketio.emit('update_terminal', {
            'output': f"[CONFIG] Chargement presets depuis {config_path} — {len(preset_options)} preset(s) trouvé(s)\n"
        }, broadcast=True)
    except Exception:
        pass
    
    return preset_options


@app.callback(
    [
        Output('look_back_saved', 'options'),
        Output('look_back_saved', 'value'),
        Output('stride_saved', 'options'),
        Output('stride_saved', 'value'),
        Output('nb_y_saved', 'options'),
        Output('nb_y_saved', 'value'),
        Output('nb_units_saved', 'options'),
        Output('nb_units_saved', 'value'),
        Output('layers_saved', 'options'),
        Output('layers_saved', 'value'),
        Output('learning_rate_saved', 'options'),
        Output('learning_rate_saved', 'value'),
        Output('loss_saved', 'options'),
        Output('loss_saved', 'value'),
        Output('epochs_saved', 'options'),
        Output('epochs_saved', 'value'),
        Output('training_days_saved', 'options'),
        Output('training_days_saved', 'value'),
        Output('train_test_ratio_saved', 'options'),
        Output('train_test_ratio_saved', 'value'),
        Output('trade_volume_saved', 'options'),
        Output('trade_volume_saved', 'value'),
        Output('k_trades_saved', 'options'),
        Output('k_trades_saved', 'value'),
        Output('model_type_saved', 'options'),
        Output('model_type_saved', 'value'),
        Output('transformer_num_heads_saved', 'options'),
        Output('transformer_num_heads_saved', 'value'),
        Output('transformer_dropout_saved', 'options'),
        Output('transformer_dropout_saved', 'value'),
        Output('transformer_ff_multiplier_saved', 'options'),
        Output('transformer_ff_multiplier_saved', 'value'),
        Output('loss_category', 'value'),
    ],
    [Input('preset_dropdown', 'id'), Input('config_bootstrap', 'n_intervals')]
)
def populate_saved_options(_, __):
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conf = {}

    def to_options(seq):
        return [{'label': str(v), 'value': v} for v in (seq or [])]

    def to_value(seq):
        # valeur par défaut = première entrée si dispo
        return seq[0] if isinstance(seq, list) and len(seq) > 0 else None

    look_back_list = conf.get('look_back_options', [])
    stride_list = conf.get('stride_options', [])
    nb_y_list = conf.get('nb_y_options', [])
    nb_units_list = conf.get('nb_units_options', [])
    layers_list = conf.get('layers_options', [])
    lr_list = conf.get('learning_rate_options', [])
    loss_list = conf.get('loss_options', [])
    epochs_list = conf.get('epochs_options', [])
    training_days_list = conf.get('training_days_options', [])
    ratio_list = conf.get('train_test_ratio_options', [])
    trade_volume_list = conf.get('trade_volume_options', [])
    k_trades_list = conf.get('k_trades_options', [])
    model_type_list = conf.get('model_type_options', [])
    heads_list = conf.get('transformer_num_heads_options', [])
    dropout_list = conf.get('transformer_dropout_options', [])
    ffmul_list = conf.get('transformer_ff_multiplier_options', [])

    look_back_opts = to_options(look_back_list)
    stride_opts = to_options(stride_list)
    nb_y_opts = to_options(nb_y_list)
    nb_units_opts = to_options(nb_units_list)
    layers_opts = to_options(layers_list)
    learning_rate_opts = to_options(lr_list)
    loss_opts = to_options(loss_list)
    epochs_opts = to_options(epochs_list)
    training_days_opts = to_options(training_days_list)
    train_test_ratio_opts = to_options(ratio_list)
    trade_volume_opts = to_options(trade_volume_list)
    k_trades_opts = to_options(k_trades_list)
    model_type_opts = [{'label': {'lstm': 'LSTM', 'gru': 'GRU', 'transformer': 'Transformer'}.get(v, v), 'value': v} for v in (model_type_list or [])]
    heads_opts = [{'label': str(v), 'value': v} for v in (heads_list or [])]
    dropout_opts = [{'label': f"{float(v):.1f}", 'value': float(v)} for v in (dropout_list or [])]
    ffmul_opts = [{'label': str(v), 'value': v} for v in (ffmul_list or [])]

    result = (
        look_back_opts, [to_value(look_back_list)] if look_back_list else [],
        stride_opts, [to_value(stride_list)] if stride_list else [],
        nb_y_opts, [to_value(nb_y_list)] if nb_y_list else [],
        nb_units_opts, [to_value(nb_units_list)] if nb_units_list else [],
        layers_opts, [to_value(layers_list)] if layers_list else [],
        learning_rate_opts, [to_value(lr_list)] if lr_list else [],
        loss_opts, [to_value(loss_list)] if loss_list else [],
        epochs_opts, [to_value(epochs_list)] if epochs_list else [],
        training_days_opts, [to_value(training_days_list)] if training_days_list else [],
        train_test_ratio_opts, [to_value(ratio_list)] if ratio_list else [],
        trade_volume_opts, [to_value(trade_volume_list)] if trade_volume_list else [],
        k_trades_opts, [to_value(k_trades_list)] if k_trades_list else [],
        model_type_opts, [to_value(model_type_list)] if model_type_list else [],
        heads_opts, [to_value(heads_list)] if heads_list else [],
        dropout_opts, [to_value(dropout_list)] if dropout_list else [],
        ffmul_opts, [to_value(ffmul_list)] if ffmul_list else [],
        'regression',
    )
    try:
        summary = {
            'look_back_options': len(look_back_list or []),
            'stride_options': len(stride_list or []),
            'nb_y_options': len(nb_y_list or []),
            'nb_units_options': len(nb_units_list or []),
            'layers_options': len(layers_list or []),
            'learning_rate_options': len(lr_list or []),
            'loss_options': len(loss_list or []),
            'epochs_options': len(epochs_list or []),
            'training_days_options': len(training_days_list or []),
            'train_test_ratio_options': len(ratio_list or []),
            'trade_volume_options': len(trade_volume_list or []),
            'k_trades_options': len(k_trades_list or []),
            'model_type_options': len(model_type_list or []),
            'transformer_num_heads_options': len(heads_list or []),
            'transformer_dropout_options': len(dropout_list or []),
            'transformer_ff_multiplier_options': len(ffmul_list or []),
        }
        socketio.emit('update_terminal', {
            'output': f"[CONFIG] Chargement options depuis {config_path}: {json.dumps(summary)}\n"
        }, broadcast=True)
    except Exception:
        pass
    return result


# Appliquer automatiquement la première valeur sélectionnée dans les listes "_saved" vers les champs principaux
@app.callback(
    [
        Output('look_back_x', 'value'),
        Output('stride_x', 'value'),
        Output('nb_y', 'value'),
        Output('nb_units', 'value'),
        Output('layers', 'value'),
        Output('learning_rate', 'value'),
        Output('loss_function', 'value'),
        Output('training_days_slider', 'value'),
        Output('train_test_ratio_slider', 'value'),
        Output('model_type', 'value'),
        Output('transformer_num_heads', 'value'),
        Output('transformer_dropout', 'value'),
        Output('transformer_ff_multiplier', 'value'),
    ],
    [
        Input('look_back_saved', 'value'),
        Input('stride_saved', 'value'),
        Input('nb_y_saved', 'value'),
        Input('nb_units_saved', 'value'),
        Input('layers_saved', 'value'),
        Input('learning_rate_saved', 'value'),
        Input('loss_saved', 'value'),
        Input('training_days_saved', 'value'),
        Input('train_test_ratio_saved', 'value'),
        Input('model_type_saved', 'value'),
        Input('transformer_num_heads_saved', 'value'),
        Input('transformer_dropout_saved', 'value'),
        Input('transformer_ff_multiplier_saved', 'value'),
    ],
)
def apply_saved_to_fields(look_back_v, stride_v, nb_y_v, nb_units_v, layers_v, lr_v, loss_v, days_v, ratio_v, model_type_v, heads_v, dropout_v, ffmul_v):
    def first_or_no_update(v):
        if isinstance(v, list) and len(v) > 0:
            return v[0]
        return dash.no_update
    return (
        first_or_no_update(look_back_v),
        first_or_no_update(stride_v),
        first_or_no_update(nb_y_v),
        first_or_no_update(nb_units_v),
        first_or_no_update(layers_v),
        first_or_no_update(lr_v),
        first_or_no_update(loss_v),
        first_or_no_update(days_v),
        first_or_no_update(ratio_v),
        first_or_no_update(model_type_v),
        first_or_no_update(heads_v),
        first_or_no_update(dropout_v),
        first_or_no_update(ffmul_v),
    )

class DashProgressCallback(Callback):
    def __init__(self, set_progress_func, total_epochs=None, stage='tuner', max_trials=None):
        super().__init__()
        self.set_progress = set_progress_func
        self.trials_results = []
        self.fig = go.Figure()
        self.fig.update_layout(
            title='Training and Validation Accuracy (Live)',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            template='plotly_dark'
        )
        self._train_da = []
        self._val_da = []
        self._train_loss = []
        self._val_loss = []
        self.total_epochs = total_epochs
        self.stage = stage
        self.max_trials = max_trials
        self.current_epoch = 0

    def set_stage(self, stage, total_epochs=None):
        self.stage = stage
        if total_epochs is not None:
            self.total_epochs = total_epochs
        # reset epoch buffers for new stage
        self._train_da = []
        self._val_da = []
        self.current_epoch = 0
        try:
            socketio.emit('update_terminal', {'output': f"\n[TRAIN] Passage en phase: {('Tuning' if stage == 'tuner' else 'Entraînement final')}\n"}, broadcast=True)
        except Exception:
            pass

    def on_train_begin(self, logs=None):
        self._train_da = []
        self._val_da = []
        self._train_loss = []
        self._val_loss = []
        self.current_epoch = 0
        # Progress à 0% dès le début
        progress_children = html.Div([
            html.Div("Démarrage de l'entraînement...", style={'marginBottom': '8px', 'color': '#4CAF50'}),
            html.Div([
                html.Div(style={'width': '0%', 'height': '10px', 'backgroundColor': '#4CAF50', 'transition': 'width 0.2s'}),
            ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#555', 'borderRadius': '4px', 'overflow': 'hidden'})
        ])
        # Utiliser une figure initiale en thème sombre
        try:
            dark_fig = go.Figure()
            dark_fig.update_layout(template='plotly_dark')
        except Exception:
            dark_fig = go.Figure()
        self.set_progress((progress_children, dark_fig))
        try:
            socketio.emit('update_terminal', {'output': "[TRAIN] Démarrage de l'entraînement...\n"}, broadcast=True)
            socketio.emit('update_progress', {'progress': 0}, broadcast=True)
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_da = logs.get('directional_accuracy') or logs.get('main_output_directional_accuracy')
        val_da = logs.get('val_directional_accuracy') or logs.get('val_main_output_directional_accuracy')
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if train_da is not None:
            self._train_da.append(train_da)
        if val_da is not None:
            self._val_da.append(val_da)
        if train_loss is not None:
            self._train_loss.append(train_loss)
        if val_loss is not None:
            self._val_loss.append(val_loss)
        self.current_epoch = epoch + 1

        # Recréer la figure live
        live_fig = go.Figure()
        if self._train_da or self._val_da:
            if self._train_da:
                live_fig.add_trace(go.Scatter(x=list(range(1, len(self._train_da) + 1)), y=self._train_da, name='Training Accuracy'))
            if self._val_da:
                live_fig.add_trace(go.Scatter(x=list(range(1, len(self._val_da) + 1)), y=self._val_da, name='Validation Accuracy'))
            live_fig.update_layout(title='Training and Validation Accuracy (Live)', template='plotly_dark')
        elif self._train_loss or self._val_loss:
            # Fallback sur les pertes si DA indisponible
            if self._train_loss:
                live_fig.add_trace(go.Scatter(x=list(range(1, len(self._train_loss) + 1)), y=self._train_loss, name='Training Loss'))
            if self._val_loss:
                live_fig.add_trace(go.Scatter(x=list(range(1, len(self._val_loss) + 1)), y=self._val_loss, name='Validation Loss'))
            live_fig.update_layout(title='Training and Validation Loss (Live)', template='plotly_dark')

        # Barre de progression
        percent = 0
        if self.total_epochs and self.total_epochs > 0:
            percent = int((self.current_epoch / self.total_epochs) * 100)

        stage_label = 'Tuning' if self.stage == 'tuner' else 'Entraînement final'
        progress_children = html.Div([
            html.Div(f"{stage_label}: epoch {self.current_epoch}/{self.total_epochs or '?'} — Train DA: {train_da}, Val DA: {val_da}", style={'marginBottom': '8px', 'color': '#4CAF50'}),
            html.Div([
                html.Div(style={'width': f'{percent}%', 'height': '10px', 'backgroundColor': '#4CAF50', 'transition': 'width 0.2s'}),
            ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#555', 'borderRadius': '4px', 'overflow': 'hidden'})
        ])

        self.set_progress((progress_children, live_fig))
        try:
            msg = f"[EPOCH {self.current_epoch}/{self.total_epochs or '?'}] train_DA={train_da} val_DA={val_da} loss={train_loss} val_loss={val_loss}\n"
            socketio.emit('update_terminal', {'output': msg}, broadcast=True)
            socketio.emit('update_progress', {'progress': percent}, broadcast=True)
        except Exception:
            pass

    def on_trial_end(self, trial):
        import pandas as pd
        trial_data = {
            'trial_id': trial.trial_id,
            'hyperparameters': str(trial.hyperparameters.values),
            'score': trial.score,
            'status': trial.status
        }
        self.trials_results.append(trial_data)

        results_df = pd.DataFrame(self.trials_results)
        results_df.rename(columns={
            'trial_id': 'Essai', 'hyperparameters': 'Hyperparamètres',
            'score': 'Score', 'status': 'Statut'
        }, inplace=True)

        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in results_df.columns],
            data=results_df.to_dict('records'),
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
        )

        # Ajouter une barre de progression (estimation) si max_trials est connu
        percent = 0
        if getattr(self, 'max_trials', None):
            num_done = len(self.trials_results)
            percent = int((num_done / self.max_trials) * 100) if self.max_trials else 0

        progress_children = html.Div([
            html.Div(f'Recherche des hyperparamètres: {len(self.trials_results)}/{getattr(self, "max_trials", "?")}', style={'marginBottom': '8px', 'color': '#4CAF50'}),
            html.Div([
                html.Div(style={'width': f'{percent}%', 'height': '10px', 'backgroundColor': '#4CAF50', 'transition': 'width 0.2s'}),
            ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#555', 'borderRadius': '4px', 'overflow': 'hidden'}),
            html.Div(table, style={'marginTop': '10px'})
        ])

        self.set_progress((progress_children, self.fig))
        try:
            socketio.emit('update_terminal', {'output': f"[TUNER] Fin essai {trial.trial_id} score={trial.score} statut={trial.status}\n"}, broadcast=True)
        except Exception:
            pass


@app.callback(
    [
        Output('training_config_status', 'children'),
        Output('preset_dropdown', 'options')
    ],
    [Input('save_training_preset', 'n_clicks'), Input('edit_training_preset', 'n_clicks')],
    [
        State('preset_name_input', 'value'),
        State('train_share_list', 'value'),
        State('look_back_saved', 'value'),
        State('stride_saved', 'value'),
        State('nb_y_saved', 'value'),
        State('nb_units_saved', 'value'),
        State('layers_saved', 'value'),
        State('learning_rate_saved', 'value'),
        State('loss_saved', 'value'),
        State('epochs_saved', 'value'),
        State('training_days_slider', 'value'),
        State('train_test_ratio_slider', 'value'),
        State('loss_category', 'value'),
        State('trade_volume_saved', 'value'),
        State('k_trades_saved', 'value'),
        State('model_type_saved', 'value'),
        State('transformer_num_heads_saved', 'value'),
        State('transformer_dropout_saved', 'value'),
        State('transformer_ff_multiplier_saved', 'value'),
    ]
)
def save_or_edit_training_preset(n_save, n_edit, preset_name, selected_train_shares, look_backs, strides, nb_y_list, nb_units_list, layers_list, lr_list, loss_list, epochs_list, training_days, train_test_ratio, loss_category, trade_volume_list, k_trades_list, model_type_list, heads_list, dropout_list, ffmul_list):
    triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0] if dash.callback_context.triggered else None
    if (not n_save and not n_edit) or not preset_name:
        if not preset_name:
            return 'Veuillez nommer le preset', []
        return '', []
        
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))
    
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conf = {}
    
    if 'presets' not in conf:
        conf['presets'] = {}

    # Si 'Modifier', remplacer le preset existant, sinon créer/écraser
    conf['presets'][preset_name] = {
        'train_share_list': selected_train_shares or [],
        'look_back_saved': look_backs or [],
        'stride_saved': strides or [],
        'nb_y_saved': nb_y_list or [],
        'nb_units_saved': nb_units_list or [],
        'layers_saved': layers_list or [],
        'learning_rate_saved': lr_list or [],
        'loss_saved': loss_list or [],
        'epochs_saved': epochs_list or [],
        'training_days': training_days,
        'train_test_ratio': train_test_ratio,
        'loss_category': loss_category,
        'trade_volume_saved': trade_volume_list or [],
        'k_trades_saved': k_trades_list or [],
        'model_type_saved': model_type_list or [],
        'transformer_num_heads_saved': heads_list or [],
        'transformer_dropout_saved': dropout_list or [],
        'transformer_ff_multiplier_saved': ffmul_list or []
    }

    with open(config_path, 'w') as f:
        json.dump(conf, f, indent=2)
    
    preset_options = [{'label': name, 'value': name} for name in conf['presets'].keys()]
    
    action = 'modifié' if triggered == 'edit_training_preset' else 'sauvegardé'
    return f"Preset '{preset_name}' {action}", preset_options


@app.callback(
    [
        Output('training_config_status', 'children'),
        Output('preset_dropdown', 'options')
    ],
    Input('delete_training_preset', 'n_clicks'),
    [
        State('preset_dropdown', 'value')
    ]
)
def delete_training_preset(n, preset_name):
    if not n or not preset_name:
        return '', []

    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conf = {}

    presets = conf.get('presets', {})
    if preset_name in presets:
        del presets[preset_name]
        conf['presets'] = presets
        with open(config_path, 'w') as f:
            json.dump(conf, f, indent=2)
        status = f"Preset '{preset_name}' supprimé"
    else:
        status = f"Preset '{preset_name}' introuvable"

    preset_options = [{'label': name, 'value': name} for name in presets.keys()]
    return status, preset_options

@app.callback(
    [
        Output('look_back_saved', 'value'),
        Output('stride_saved', 'value'),
        Output('nb_y_saved', 'value'),
        Output('nb_units_saved', 'value'),
        Output('layers_saved', 'value'),
        Output('learning_rate_saved', 'value'),
        Output('loss_saved', 'value'),
        Output('epochs_saved', 'value'),
        Output('train_share_list', 'value'),
        Output('training_days_slider', 'value'),
        Output('train_test_ratio_slider', 'value'),
        Output('loss_category', 'value'),
        Output('trade_volume_saved', 'value'),
        Output('k_trades_saved', 'value'),
        Output('model_type_saved', 'value'),
        Output('transformer_num_heads_saved', 'value'),
        Output('transformer_dropout_saved', 'value'),
        Output('transformer_ff_multiplier_saved', 'value'),
        Output('training_config_status', 'children')
    ],
    Input('load_training_preset', 'n_clicks'),
    [State('preset_dropdown', 'value')]
)
def load_training_preset(n, preset_name):
    if not n or not preset_name:
        return [], [], [], [], [], [], [], [], [], None, None, None, [], [], [], [], [], [], ''
        
    config_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'config_data.json'))
    
    try:
        with open(config_path, 'r') as f:
            conf = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], [], [], [], [], [], [], [], None, None, [], [], [], [], 'Aucun preset trouvé'

    preset = conf.get('presets', {}).get(preset_name)

    if not preset:
        return [], [], [], [], [], [], [], [], [], None, None, None, [], [], [], [], [], [], f"Preset '{preset_name}' non trouvé"

    return (
        preset.get('look_back_saved', []),
        preset.get('stride_saved', []),
        preset.get('nb_y_saved', []),
        preset.get('nb_units_saved', []),
        preset.get('layers_saved', []),
        preset.get('learning_rate_saved', []),
        preset.get('loss_saved', []),
        preset.get('epochs_saved', []),
        preset.get('train_share_list', []),
        preset.get('training_days', None),
        preset.get('train_test_ratio', None),
        preset.get('loss_category', None),
        preset.get('trade_volume_saved', []),
        preset.get('k_trades_saved', []),
        preset.get('model_type_saved', []),
        preset.get('transformer_num_heads_saved', []),
        preset.get('transformer_dropout_saved', []),
        preset.get('transformer_ff_multiplier_saved', []),
        f"Preset '{preset_name}' chargé"
    )


@app.callback(
    [
        Output('training-results', 'children'),
        Output('model_metrics', 'children')
    ],
    [
        Input('train_button', 'n_clicks')
    ],
    [
        State('train_share_list', 'value'),
        State('training_days_slider', 'value'),
        State('train_test_ratio_slider', 'value'),
        State('tuning_method', 'value'),
        State('trade_volume', 'value'),
        State('k_trades', 'value'),
        State('look_back_x', 'value'),
        State('stride_x', 'value'),
        State('nb_y', 'value'),
        State('nb_units', 'value'),
        State('layers', 'value'),
        State('learning_rate', 'value'),
        State('loss_function', 'value'),
        State('model_type', 'value'),
        State('transformer_num_heads', 'value'),
        State('transformer_dropout', 'value'),
        State('transformer_ff_multiplier', 'value'),
        State('epochs', 'value'),
        State('look_back_saved', 'value'),
        State('stride_saved', 'value'),
        State('nb_y_saved', 'value'),
        State('nb_units_saved', 'value'),
        State('layers_saved', 'value'),
        State('learning_rate_saved', 'value'),
        State('loss_saved', 'value'),
        State('model_type_saved', 'value'),
        State('transformer_num_heads_saved', 'value'),
        State('transformer_dropout_saved', 'value'),
        State('transformer_ff_multiplier_saved', 'value'),
    ],
    background=True,
    progress=[
        Output('training_progress', 'children'),
        Output('accuracy_graph', 'figure')
    ],
    running=[
        (Output('train_button', 'disabled'), True, False)
    ],
    prevent_initial_call=True
)
def train_and_display_progress(set_progress, n_clicks, selected_symbols, training_days,
                               train_test_ratio, tuning_method, trade_volume, k_trades, look_back_x, stride_x, nb_y,
                               nb_units, layers, learning_rate, loss_function, model_type, heads, dropout, ffmul, epochs,
                               look_back_opts, stride_opts, nb_y_opts,
                               nb_units_opts, layers_opts, learning_rate_opts, loss_opts,
                               model_type_opts_saved, heads_opts_saved, dropout_opts_saved, ffmul_opts_saved):
    # Validation champs requis selon architecture
    missing = []
    if not selected_symbols:
        missing.append('Action(s) à entraîner')
    if not (training_days and isinstance(training_days, int)):
        missing.append('Nombre de jours de données')
    if not (train_test_ratio and isinstance(train_test_ratio, int)):
        missing.append('Ratio Entraînement/Test')
    if not (look_back_x and look_back_x > 0):
        missing.append('Look Back (X)')
    if not (stride_x and stride_x > 0):
        missing.append('Stride (X)')
    if not (nb_y and nb_y > 0):
        missing.append('Nombre de Y')
    if not (nb_units):
        missing.append("Nombre d'Unités")
    if not (layers):
        missing.append('Nombre de Couches')
    if not (learning_rate):
        missing.append("Taux d'Apprentissage")
    if not (loss_function or (loss_opts and len(loss_opts) > 0)):
        missing.append('Fonction(s) de Perte')
    if not (model_type):
        missing.append('Type de Modèle')
    if model_type == 'transformer':
        if not (heads or (heads_opts_saved and len(heads_opts_saved) > 0)):
            missing.append("Têtes d'Attention (Transformer)")
        if dropout is None and not (dropout_opts_saved and len(dropout_opts_saved) > 0):
            missing.append('Dropout')
        if not (ffmul or (ffmul_opts_saved and len(ffmul_opts_saved) > 0)):
            missing.append('Facteur Feed-Forward (Transformer)')
    if not (trade_volume and trade_volume > 0):
        missing.append('Volume de vente (par trade)')
    if not (k_trades and k_trades > 0):
        missing.append('Nombre de K (trades max/jour)')

    if missing:
        modal = html.Div([
            html.Div([
                html.Div('Paramètres manquants', style={'fontWeight': 'bold', 'marginBottom': '8px', 'color': '#FF8C00'}),
                html.Ul([html.Li(m) for m in missing], style={'textAlign': 'left', 'color': '#FFFFFF'}),
            ], style={'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px', 'border': '1px solid #555', 'maxWidth': '520px', 'margin': '0 auto'})
        ])
        return modal, dash.no_update

    # Filtrer l'option "All" si présente et garder une action réelle
    symbols = [s for s in selected_symbols or [] if s != 'All']
    if not symbols:
        return html.P("Veuillez sélectionner une action spécifique."), "Sélectionnez une action."
    symbol = symbols[0]
    share_series = shM.getRowsDfByKeysValues(['symbol'], [symbol])
    if share_series.empty:
        return html.P(f"Aucune donnée pour {symbol}"), f"Erreur: {symbol} non trouvé."
    shareObj = share_series.iloc[0]

    # Utiliser les valeurs des dropdowns multi-sélection s'ils sont remplis
    layers_list = layers_opts if layers_opts else [layers]
    nb_units_list = nb_units_opts if nb_units_opts else [nb_units]
    learning_rate_list = learning_rate_opts if learning_rate_opts else [learning_rate]
    loss_list = loss_opts if loss_opts else [loss_function]
    architecture_list = model_type_opts_saved if model_type_opts_saved else [model_type or 'lstm']
    heads_list = heads_opts_saved if heads_opts_saved else ([heads] if heads is not None else [4])
    dropout_list = dropout_opts_saved if dropout_opts_saved else ([float(dropout)] if dropout is not None else [0.1])
    ffmul_list = ffmul_opts_saved if ffmul_opts_saved else ([ffmul] if ffmul is not None else [4])
    
    hps = {
        'layers': layers_list,
        'nb_units': nb_units_list,
        'learning_rate': learning_rate_list,
        'loss': loss_list,
        'architecture': architecture_list,
        'transformer_num_heads': heads_list,
        'transformer_dropout': dropout_list,
        'transformer_ff_multiplier': ffmul_list,
        'tuning_method': tuning_method or 'random',
        'max_trials': 5,
        'executions_per_trial': 1,
        'directory': 'tuner_results_ui',
        'project_name': f'pred_{symbol}_ui',
        'patience': 3,
        'epochs': epochs or 15
    }

    # Utiliser les valeurs des dropdowns multi-sélection pour data_info
    data_info = {
        'look_back_x': min(look_back_opts) if look_back_opts else look_back_x,
        'stride_x': min(stride_opts) if stride_opts else stride_x,
        'nb_y': min(nb_y_opts) if nb_y_opts else nb_y,
        'features': ['openPrice', 'volume'], 'return_type': 'yield',
        'nb_days_to_take_dataset': training_days, 'percent_train_test': train_test_ratio,
        'shareObj': shareObj,
        'trade_volume': trade_volume,
        'k_trades': k_trades
    }

    try:
        logging.info(f"[TRAIN] Start: symbol={symbol}, days={training_days}, split={train_test_ratio}, look_back={look_back_x}, stride_x={stride_x}, nb_y={nb_y}, layers={layers}, nb_units={nb_units}, lr={learning_rate}, loss={loss_function}")
        # Mise à jour immédiate de l'UI pour indiquer le démarrage + barre de progression à 0%
        progress_children = html.Div([
            html.Div("Initialisation de l'entraînement...", style={'marginBottom': '8px', 'color': '#4CAF50'}),
            html.Div([
                html.Div(style={'width': '0%', 'height': '10px', 'backgroundColor': '#4CAF50', 'transition': 'width 0.2s'}),
            ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#555', 'borderRadius': '4px', 'overflow': 'hidden'})
        ])
        # Figure initiale en thème sombre
        try:
            dark_fig = go.Figure()
            dark_fig.update_layout(template='plotly_dark')
        except Exception:
            dark_fig = go.Figure()
        set_progress((progress_children, dark_fig))
        try:
            params_dump = {
                'symbol': symbol,
                'tuning_method': tuning_method,
                'data_info': {
                    'look_back_x': data_info['look_back_x'],
                    'stride_x': data_info['stride_x'],
                    'nb_y': data_info['nb_y'],
                    'features': data_info['features'],
                    'return_type': data_info['return_type'],
                    'training_days': data_info['nb_days_to_take_dataset'],
                    'train_test_ratio': data_info['percent_train_test'],
                    'trade_volume': data_info['trade_volume'],
                    'k_trades': data_info['k_trades']
                },
                'hps': {
                    'architecture': hps['architecture'],
                    'layers': hps['layers'],
                    'nb_units': hps['nb_units'],
                    'learning_rate': hps['learning_rate'],
                    'loss': hps['loss'],
                    'transformer_num_heads': hps['transformer_num_heads'],
                    'transformer_dropout': hps['transformer_dropout'],
                    'transformer_ff_multiplier': hps['transformer_ff_multiplier'],
                    'epochs': hps['epochs'],
                    'max_trials': hps.get('max_trials'),
                    'executions_per_trial': hps.get('executions_per_trial')
                }
            }
            socketio.emit('update_terminal', {'output': f"[TRAIN] Lancement: {symbol} (jours={training_days}, split={train_test_ratio}%)\n[PARAMS] {json.dumps(params_dump)}\n"}, broadcast=True)
        except Exception:
            pass
        dash_callback = DashProgressCallback(set_progress, total_epochs=hps['epochs'], stage='tuner', max_trials=hps['max_trials'])
        logging.info("[TRAIN] Tuner: running search")
        try:
            dark_fig = go.Figure()
            dark_fig.update_layout(template='plotly_dark')
        except Exception:
            dark_fig = go.Figure()
        set_progress((html.Div("Tuning des hyperparamètres...", style={'color': '#4CAF50'}), dark_fig))
        try:
            socketio.emit('update_terminal', {'output': "[TUNER] Démarrage de la recherche...\n"}, broadcast=True)
        except Exception:
            pass
        best_model, best_hps, tuner = shM.train_share_model(shareObj, data_info, hps, callbacks=[dash_callback])
        logging.info("[TRAIN] Tuner: done")
        try:
            socketio.emit('update_terminal', {'output': "[TUNER] Recherche terminée.\n"}, broadcast=True)
        except Exception:
            pass

        # Passer en mode entraînement final (progression par epoch)
        if hasattr(dash_callback, 'set_stage'):
            dash_callback.set_stage('final', total_epochs=hps['epochs'])
        logging.info("[TRAIN] Fit: start")
        set_progress((html.Div("Entraînement final...", style={'color': '#4CAF50'}), go.Figure()))
        try:
            socketio.emit('update_terminal', {'output': "[FIT] Démarrage de l'entraînement final...\n"}, broadcast=True)
        except Exception:
            pass

        # Garde: vérifier qu'il y a au moins une séquence
        if getattr(tuner, 'trainX', None) is None or len(tuner.trainX) == 0:
            raise ValueError("Aucune séquence d'entraînement disponible (trainX vide)")
        if getattr(tuner, 'testX', None) is None or len(tuner.testX) == 0:
            raise ValueError("Aucune séquence de validation disponible (testX vide)")

        history = best_model.fit(
            tuner.trainX,
            tuner.trainY,
            epochs=hps['epochs'],
            validation_data=(tuner.testX, tuner.testY),
            callbacks=[dash_callback]
        )
        logging.info("[TRAIN] Fit: done")
        try:
            socketio.emit('update_terminal', {'output': "[FIT] Entraînement final terminé.\n"}, broadcast=True)
        except Exception:
            pass
        final_fig = go.Figure()
        # Récupérer les métriques selon les noms disponibles
        train_da = history.history.get('directional_accuracy') or history.history.get('main_output_directional_accuracy')
        val_da = history.history.get('val_directional_accuracy') or history.history.get('val_main_output_directional_accuracy')
        if train_da is not None:
            final_fig.add_trace(go.Scatter(x=list(range(1, len(train_da)+1)), y=train_da, name='Training Accuracy'))
        if val_da is not None:
            final_fig.add_trace(go.Scatter(x=list(range(1, len(val_da)+1)), y=val_da, name='Validation Accuracy'))
        if train_da is None and val_da is None:
            # Fallback sur les pertes
            train_loss = history.history.get('loss')
            val_loss = history.history.get('val_loss')
            if train_loss is not None:
                final_fig.add_trace(go.Scatter(x=list(range(1, len(train_loss)+1)), y=train_loss, name='Training Loss'))
            if val_loss is not None:
                final_fig.add_trace(go.Scatter(x=list(range(1, len(val_loss)+1)), y=val_loss, name='Validation Loss'))
        final_fig.update_layout(title=f'Résultats finaux pour {symbol}', template='plotly_dark')
        # Pousser le graphique final dans le flux de progression pour l'afficher
        try:
            set_progress((html.Div("Entraînement final terminé.", style={'color': '#4CAF50'}), final_fig))
        except Exception:
            pass

        # Sauvegarder le modèle uniquement après l'entraînement final
        try:
            best_trial = tuner.oracle.get_best_trials(1)[0]
            status_message = f"Entraînement terminé. Meilleur score: {best_trial.score:.4f}"
            # Calculer des scores moyens si dispo
            try:
                import numpy as np
                da_hist = best_trial.metrics.get_history('directional_accuracy')
                val_da_hist = best_trial.metrics.get_history('val_directional_accuracy')
                trainScore = float(np.mean([m.value for m in da_hist])) if da_hist else None
                testScore = float(np.mean([m.value for m in val_da_hist])) if val_da_hist else None
            except Exception:
                trainScore = None
                testScore = None
            model_name = hps.get('project_name', f'pred_{symbol}_ui')
            shM.save_model(shareObj, best_model, model_name, trainScore, testScore)
        except Exception as e:
            logging.exception("[TRAIN] Save after final fit failed")
            status_message = f"Entraînement terminé. (Erreur sauvegarde: {e})"
        logging.info("[TRAIN] Success: %s", status_message)
        try:
            socketio.emit('update_terminal', {'output': f"[SUCCESS] {status_message}\n"}, broadcast=True)
        except Exception:
            pass

        return status_message, dash_table.DataTable(data=[], columns=[])

    except Exception as e:
        logging.exception("[TRAIN] Failure")
        try:
            socketio.emit('update_terminal', {'output': f"[ERROR] {str(e)}\n"}, broadcast=True)
        except Exception:
            pass
        error_msg = html.Div([
            html.H5("Erreur lors de l'entraînement"),
            html.Pre(str(e)),
            html.Pre(traceback.format_exc())
        ])
        set_progress((html.Div(f"Erreur: {str(e)}", style={'color': '#f44336'}), go.Figure()))
        return error_msg, None

"""
Note: les métriques de performance du graphique sont mises à jour dans
`prediction_callbacks.graph.update_performance_metrics`.
"""


