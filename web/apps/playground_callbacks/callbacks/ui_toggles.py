# -*- coding: utf-8 -*-
"""
Callbacks de toggle UI pour le playground.
"""

from dash import Input, Output, html
from app import app


@app.callback(
    [
        Output('panel_play_new', 'style'),
        Output('panel_play_saved', 'style'),
        Output('panel_play_btn_new', 'style'),
        Output('panel_play_btn_saved', 'style'),
        Output('panel_model_type_selector', 'style'),
        Output('panel_play_data_params', 'style'),
        Output('label_lstm_params', 'style'),
        Output('panel_lstm_params', 'style'),
        Output('label_transformer_params', 'style'),
        Output('panel_transformer_params', 'style'),
        Output('label_hybrid_params', 'style'),
        Output('panel_hybrid_params', 'style'),
    ],
    [Input('play_model_mode', 'value')]
)
def toggle_play_panels(mode):
    """Toggle entre mode nouveau modèle et modèle sauvegardé."""
    show_grid = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px'}
    show_data_grid = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(130px, 1fr))', 'gap': '8px'}
    hide = {'display': 'none'}
    show_btn = {'display': 'block'}
    show_block = {'display': 'block', 'marginBottom': '12px'}
    show_label = {'display': 'block'}
    
    if mode == 'saved':
        return (
            hide,  # panel_play_new
            {'display': 'block'},  # panel_play_saved
            hide,  # panel_play_btn_new
            show_btn,  # panel_play_btn_saved
            hide,  # panel_model_type_selector
            hide,  # panel_play_data_params
            hide,  # label_lstm_params
            hide,  # panel_lstm_params
            hide,  # label_transformer_params
            hide,  # panel_transformer_params
            hide,  # label_hybrid_params
            hide,  # panel_hybrid_params
        )
    
    return (
        show_grid,  # panel_play_new
        hide,  # panel_play_saved
        show_btn,  # panel_play_btn_new
        hide,  # panel_play_btn_saved
        show_block,  # panel_model_type_selector
        show_data_grid,  # panel_play_data_params
        show_label,  # label_lstm_params
        show_grid,  # panel_lstm_params
        hide,  # label_transformer_params
        hide,  # panel_transformer_params
        hide,  # label_hybrid_params
        hide,  # panel_hybrid_params
    )


@app.callback(
    [
        Output('label_lstm_params', 'style', allow_duplicate=True),
        Output('panel_lstm_params', 'style', allow_duplicate=True),
        Output('label_transformer_params', 'style', allow_duplicate=True),
        Output('panel_transformer_params', 'style', allow_duplicate=True),
        Output('label_hybrid_params', 'style', allow_duplicate=True),
        Output('panel_hybrid_params', 'style', allow_duplicate=True),
    ],
    [Input('play_model_type', 'value')],
    prevent_initial_call=True
)
def toggle_model_type_params(model_type):
    """Affiche les paramètres correspondant au type de modèle sélectionné."""
    show_grid = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px'}
    show_grid_lstm = {'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px'}
    hide = {'display': 'none'}
    show_label = {'display': 'block'}
    
    if model_type == 'transformer':
        return hide, hide, show_label, show_grid, hide, hide
    elif model_type == 'hybrid':
        return hide, hide, hide, hide, show_label, show_grid
    else:  # lstm par défaut
        return show_label, show_grid_lstm, hide, hide, hide, hide


@app.callback(
    Output('curve_info_msg', 'children'),
    [Input('play_curve_type', 'value')]
)
def update_curve_info_message(curve_type):
    """Affiche un message d'aide selon le type de courbe sélectionné."""
    messages = {
        'random_walk': html.Div([
            html.Span('🎲 ', style={'fontSize': '16px'}),
            html.Span('Random Walk : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Marche aléatoire. ', style={'color': '#888'}),
            html.Span('Bruit', style={'color': '#4CAF50'}),
            html.Span(' contrôle l\'amplitude des variations.', style={'color': '#888'}),
        ]),
        'trend': html.Div([
            html.Span('📈 ', style={'fontSize': '16px'}),
            html.Span('Trend : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Tendance + bruit. ', style={'color': '#888'}),
            html.Span('Trend > 0', style={'color': '#4CAF50'}),
            html.Span(' = hausse, ', style={'color': '#888'}),
            html.Span('< 0', style={'color': '#f44336'}),
            html.Span(' = baisse.', style={'color': '#888'}),
        ]),
        'seasonal': html.Div([
            html.Span('🌊 ', style={'fontSize': '16px'}),
            html.Span('Seasonal : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Cycle sinusoïdal journalier + bruit. ', style={'color': '#888'}),
            html.Span('Amplitude', style={'color': '#4CAF50'}),
            html.Span(' = force du cycle.', style={'color': '#888'}),
        ]),
        'lunch_effect': html.Div([
            html.Span('🍽️ ', style={'fontSize': '16px'}),
            html.Span('Lunch Effect : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Baisse prix 12h-14h + bruit. ', style={'color': '#888'}),
            html.Span('Lunch effect', style={'color': '#4CAF50'}),
            html.Span(' = intensité de la baisse.', style={'color': '#888'}),
        ]),
        'sinusoidale': html.Div([
            html.Span('〰️ ', style={'fontSize': '16px'}),
            html.Span('Sinusoïdale : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Oscillation régulière. ', style={'color': '#888'}),
            html.Span('Période', style={'color': '#4CAF50'}),
            html.Span(' = durée cycle, ', style={'color': '#888'}),
            html.Span('Bruit=0', style={'color': '#4CAF50'}),
            html.Span(' = parfait.', style={'color': '#888'}),
        ]),
        'plateau': html.Div([
            html.Span('📊 ', style={'fontSize': '16px'}),
            html.Span('Plateau : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('N niveaux aléatoires répétés. ', style={'color': '#888'}),
            html.Span('Bruit=0', style={'color': '#4CAF50'}),
            html.Span(' = déterministe. ', style={'color': '#888'}),
            html.Span('Idéal pour tester l\'IA !', style={'color': '#2196F3', 'fontWeight': 'bold'}),
        ]),
        'pattern': html.Div([
            html.Span('🧩 ', style={'fontSize': '16px'}),
            html.Span('Pattern : ', style={'color': '#FF8C00', 'fontWeight': 'bold'}),
            html.Span('Amorce (K min) + motif. ', style={'color': '#888'}),
            html.Span('Relations N:M', style={'color': '#4CAF50'}),
            html.Span(' entre amorces et patterns. ', style={'color': '#888'}),
            html.Span('Teste la reconnaissance de séquences !', style={'color': '#2196F3', 'fontWeight': 'bold'}),
        ]),
    }
    return messages.get(curve_type, html.Div())

