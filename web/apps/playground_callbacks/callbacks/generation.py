# -*- coding: utf-8 -*-
"""
Callbacks de génération de courbes synthétiques.
"""

import dash
from dash import Input, Output, State, html
import plotly.graph_objs as go
import pandas as pd
from io import StringIO

from app import app
from web.services.synthetic import generate_synthetic_timeseries
from web.apps.model_config import DEFAULT_LOOK_BACK


@app.callback(
    [
        Output('play_patterns_par_amorce', 'max'),
        Output('play_amorces_par_pattern', 'max'),
        Output('play_nb_patterns', 'value'),
        Output('play_nb_amorces', 'value'),
        Output('play_patterns_par_amorce', 'value'),
        Output('play_amorces_par_pattern', 'value'),
        Output('play_pattern_ratio_pref', 'data'),
    ],
    [
        Input('play_nb_patterns', 'value'),
        Input('play_nb_amorces', 'value'),
        Input('play_patterns_par_amorce', 'value'),
        Input('play_amorces_par_pattern', 'value'),
    ],
    [
        State('play_pattern_ratio_pref', 'data'),
    ],
    prevent_initial_call=True,
)
def sync_pattern_params(nb_patterns, nb_amorces, patterns_par_amorce, amorces_par_pattern, ratio_pref):
    """
    Synchronise automatiquement:
    - patterns_par_amorce <= nb_patterns
    - amorces_par_pattern <= nb_amorces

    Règle demandée:
    - si l'utilisateur change un ratio, on ajuste #patterns/#amorces à la hausse si possible
    - si l'utilisateur change #patterns/#amorces, on plafonne les ratios
    """
    def _to_int(v, default):
        try:
            if v is None:
                return int(default)
            return int(v)
        except Exception:
            return int(default)

    nb_patterns = _to_int(nb_patterns, 4)
    nb_amorces = _to_int(nb_amorces, 4)
    patterns_par_amorce = _to_int(patterns_par_amorce, 2)
    amorces_par_pattern = _to_int(amorces_par_pattern, 2)

    # bornes UI (doivent matcher le layout)
    NB_MAX = 20
    RATIO_MAX = 10

    nb_patterns = max(1, min(nb_patterns, NB_MAX))
    nb_amorces = max(1, min(nb_amorces, NB_MAX))
    patterns_par_amorce = max(1, min(patterns_par_amorce, RATIO_MAX))
    amorces_par_pattern = max(1, min(amorces_par_pattern, RATIO_MAX))

    # Préférences (ratios "souhaités") mémorisées
    if not isinstance(ratio_pref, dict):
        ratio_pref = {}
    pref_patterns_par_amorce = _to_int(ratio_pref.get('patterns_par_amorce'), patterns_par_amorce)
    pref_amorces_par_pattern = _to_int(ratio_pref.get('amorces_par_pattern'), amorces_par_pattern)
    pref_patterns_par_amorce = max(1, min(pref_patterns_par_amorce, RATIO_MAX))
    pref_amorces_par_pattern = max(1, min(pref_amorces_par_pattern, RATIO_MAX))

    triggered = None
    try:
        if dash.callback_context.triggered:
            triggered = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    except Exception:
        triggered = None

    # Si l'utilisateur modifie le ratio -> mettre à jour la préférence puis monter nb_* si nécessaire
    if triggered in ('play_patterns_par_amorce', 'play_amorces_par_pattern'):
        pref_patterns_par_amorce = patterns_par_amorce
        pref_amorces_par_pattern = amorces_par_pattern
        if patterns_par_amorce > nb_patterns:
            nb_patterns = min(patterns_par_amorce, NB_MAX)
        if amorces_par_pattern > nb_amorces:
            nb_amorces = min(amorces_par_pattern, NB_MAX)

    # Si l'utilisateur modifie nb_* -> on n’écrase pas la préférence,
    # mais on recalcule le ratio effectif avec min(préférence, nb_*)
    if triggered in ('play_nb_patterns', 'play_nb_amorces'):
        patterns_par_amorce = min(pref_patterns_par_amorce, nb_patterns)
        amorces_par_pattern = min(pref_amorces_par_pattern, nb_amorces)

    # Dans tous les cas, on garantit la cohérence finale
    patterns_par_amorce = min(max(1, patterns_par_amorce), nb_patterns)
    amorces_par_pattern = min(max(1, amorces_par_pattern), nb_amorces)

    # max dynamiques (empêche de saisir > nb_*)
    max_patterns_par_amorce = nb_patterns
    max_amorces_par_pattern = nb_amorces

    return (
        max_patterns_par_amorce,
        max_amorces_par_pattern,
        nb_patterns,
        nb_amorces,
        patterns_par_amorce,
        amorces_par_pattern,
        {
            'patterns_par_amorce': pref_patterns_par_amorce,
            'amorces_par_pattern': pref_amorces_par_pattern,
        },
    )


@app.callback(
    [
        Output('container_sine_period', 'style'),
        Output('container_sine_reset', 'style'),
        Output('container_sine_offset', 'style'),
        Output('container_nb_plateaux', 'style'),
        Output('container_lunch_strength', 'style'),
        Output('play_pattern_params_container', 'style'),
    ],
    [Input('play_curve_type', 'value')],
)
def toggle_generation_params(curve_type):
    """Affiche/Masque les paramètres spécifiques selon le type de courbe."""
    hide = {'display': 'none'}
    show = {'display': 'block'}

    if curve_type == 'sinusoidale':
        return show, show, show, hide, hide, hide
    if curve_type == 'plateau':
        return hide, hide, hide, show, hide, hide
    if curve_type == 'lunch_effect':
        return hide, hide, hide, hide, show, hide
    if curve_type == 'pattern':
        return hide, hide, hide, hide, hide, show
    return hide, hide, hide, hide, hide, hide


@app.callback(
    [
        Output('play_segments_graph', 'figure'),
        Output('play_df_store', 'data'),
        Output('play_train_backtest', 'disabled'),
        Output('play_model_ready', 'data'),
        Output('play_model_path', 'data'),
        Output('play_predictions_store', 'data'),
        Output('play_gen_summary', 'children'),
    ],
    [Input('play_generate', 'n_clicks')],
    [
        State('play_curve_type', 'value'),
        State('play_date_range', 'start_date'),
        State('play_date_range', 'end_date'),
        State('play_open_time', 'value'),
        State('play_close_time', 'value'),
        State('play_base_price', 'value'),
        State('play_noise', 'value'),
        State('play_trend_strength', 'value'),
        State('play_seasonality_amp', 'value'),
        State('play_sine_period', 'value'),
        State('play_sine_reset_daily', 'value'),
        State('play_sine_phase_shift', 'value'),
        State('play_nb_plateaux', 'value'),
        State('play_lunch_strength', 'value'),
        State('play_seed', 'value'),
        # Paramètres Pattern
        State('play_nb_patterns', 'value'),
        State('play_nb_amorces', 'value'),
        State('play_duree_amorce', 'value'),
        State('play_patterns_par_amorce', 'value'),
        State('play_amorces_par_pattern', 'value'),
    ],
    prevent_initial_call=True,
)
def generate_curve(n_clicks, curve_type, start_date, end_date, open_time, close_time, 
                   base_price, noise_val, trend_s, seas_amp, sine_period, sine_reset_daily, sine_phase_shift, nb_plateaux, lunch_s, seed,
                   nb_patterns, nb_amorces, duree_amorce, patterns_par_amorce, amorces_par_pattern):
    """Génère une courbe synthétique selon les paramètres."""
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000',
        font={'color': '#FFFFFF'}, title='Série synthétique — cliquer sur Générer',
        height=420, uirevision='play_segments'
    )
    
    if not n_clicks:
        return (empty_fig, None, True, dash.no_update, dash.no_update, None,
                html.Div("Cliquez sur Générer pour créer une nouvelle courbe.", style={'color': '#CCCCCC', 'fontSize': '12px'}))
    
    try:
        # Ajustements (pour expliquer les combinaisons restrictives en mode pattern)
        warn_lines = []
        if (curve_type or '') == 'pattern':
            try:
                nb_p = int(nb_patterns) if nb_patterns is not None else 4
                nb_a = int(nb_amorces) if nb_amorces is not None else 4
                ppa = int(patterns_par_amorce) if patterns_par_amorce is not None else 2
                app = int(amorces_par_pattern) if amorces_par_pattern is not None else 2

                if ppa > nb_p:
                    warn_lines.append(f"patterns/amorce plafonné: {ppa} → {nb_p} (nb_patterns={nb_p})")
                if app > nb_a:
                    warn_lines.append(f"amorces/pattern plafonné: {app} → {nb_a} (nb_amorces={nb_a})")
            except Exception:
                pass

        df = generate_synthetic_timeseries(
            start_date, end_date,
            market_open=open_time or '09:30',
            market_close=close_time or '16:00',
            base_price=float(base_price or 100.0),
            data_type=str(curve_type or 'random_walk'),
            seed=int(seed) if seed is not None else None,
            noise=float(noise_val) if noise_val is not None else 0.0,
            trend_strength=float(trend_s) if trend_s is not None else 0.0,
            seasonality_amplitude=float(seas_amp) if seas_amp is not None else 0.0,
            lunch_effect_strength=float(lunch_s) if lunch_s is not None else 0.0,
            sine_period_minutes=int(sine_period) if sine_period is not None else 360,
            nb_plateaux=int(nb_plateaux) if nb_plateaux is not None else 3,
            sine_reset_daily=True if (sine_reset_daily and True in sine_reset_daily) else False,
            sine_phase_shift_deg=float(sine_phase_shift) if sine_phase_shift is not None else 0.0,
            # Paramètres Pattern
            nb_patterns=int(nb_patterns) if nb_patterns is not None else 4,
            nb_amorces=int(nb_amorces) if nb_amorces is not None else 4,
            duree_amorce=int(duree_amorce) if duree_amorce is not None else 60,
            patterns_par_amorce=int(patterns_par_amorce) if patterns_par_amorce is not None else 2,
            amorces_par_pattern=int(amorces_par_pattern) if amorces_par_pattern is not None else 2,
        )
        
        if df is None or df.empty:
            empty_fig.update_layout(title='Aucune donnée générée (plage/horaires vides)')
            return (empty_fig, None, True, dash.no_update, dash.no_update, None,
                    html.Div("Aucune donnée générée (plage/horaires vides)", style={'color': '#F59E0B', 'fontSize': '12px'}))
        
        # Stocker les données
        store = df[['openPrice']].to_json(date_format='iso', orient='split')
        
        # Construire le graphe
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['openPrice'], mode='lines', name='Prix',
            line={'color': '#FF8C00', 'width': 2}
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={'color': '#FFF'},
            title=f'📊 {curve_type.upper()} — {len(df)} points générés',
            height=420, uirevision='play_segments',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        base_msg = "Courbe générée. Modèle précédent toujours disponible pour la généralisation. Ré-entraîner si besoin."
        if warn_lines:
            msg = html.Div([
                html.Div(base_msg),
                html.Div(" | ".join(warn_lines), style={'color': '#F59E0B', 'marginTop': '4px'}),
            ], style={'color': '#94a3b8', 'fontSize': '12px'})
        else:
            msg = html.Div(base_msg, style={'color': '#94a3b8', 'fontSize': '12px'})
        return fig, store, False, dash.no_update, dash.no_update, None, msg
        
    except Exception as e:
        empty_fig.update_layout(title=f'Erreur génération: {e}', height=420, uirevision='play_segments')
        return (empty_fig, None, True, dash.no_update, dash.no_update, None,
                html.Div(f"Erreur génération: {e}", style={'color': '#EF4444', 'fontSize': '12px'}))


@app.callback(
    Output('play_mini_preview', 'figure'),
    [Input('play_df_store', 'data')],
    [State('play_curve_type', 'value')]
)
def update_mini_preview(store_json, curve_type):
    """Met à jour le mini preview de la courbe."""
    if not store_json:
        return {'data': [], 'layout': {'height': 1, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}}
    
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
        if df.empty:
            return {'data': [], 'layout': {'height': 1}}
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['openPrice'], mode='lines',
            line={'color': '#FF8C00', 'width': 1.5},
            hoverinfo='none'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=150,
            showlegend=False
        )
        return fig
    except Exception:
        return {'data': [], 'layout': {'height': 1}}

