from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from app import app, shM
from web.components.navigation import create_navigation, create_page_help
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import dash
import threading
from web.services.correlation import (
    get_returns_with_daily_fallback,
    corr_matrix_with_lag,
    corr_matrix_with_lag_fast,
    corr_matrix_max_0_30,
    daily_crosscorr_points,
    get_minute_returns,
)

# Etat global pour calcul incrémental du Top 10 corrélations
_topcorr_lock = threading.Lock()
_topcorr_job_state = {
    'running': False,
    'progress': 0,
    'total': 0,
    'results': {},  # (sym_i, sym_j) -> (max_corr, lag)
    'top10': [],    # liste de tuples (sym_i, sym_j, corr, lag)
    'error': None,
    'thread': None,
}


def _run_topcorr_job():
    try:
        end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=15)
        df = shM.dfShares
        symbols = [row.symbol for row in df.itertuples() if getattr(row, 'symbol', None)]
        returns = get_minute_returns(shM, symbols, start_date, end_date)
        with _topcorr_lock:
            _topcorr_job_state['error'] = None
            _topcorr_job_state['results'] = {}
            _topcorr_job_state['top10'] = []
            _topcorr_job_state['progress'] = 0
        if returns.empty or len(returns.columns) < 2:
            with _topcorr_lock:
                _topcorr_job_state['error'] = "Données minute insuffisantes pour calculer les corrélations."
                _topcorr_job_state['running'] = False
            return
        cols = list(returns.columns)
        total = (len(cols) * (len(cols) - 1)) // 2
        with _topcorr_lock:
            _topcorr_job_state['total'] = total
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                # Vérifier si annulation éventuelle (non implémentée ici)
                s_i = cols[i]
                s_j = cols[j]
                base_i = returns[s_i]
                base_j = returns[s_j]
                best_val = -np.inf
                best_k = 0
                for k in range(0, 31):
                    if k == 0:
                        right = base_j
                    else:
                        right = base_j.shift(k)
                    # Corrélation "hausses uniquement" (co-augmentations)
                    df_pair = pd.concat([base_i, right], axis=1, join='inner').dropna()
                    df_pair = df_pair[(df_pair.iloc[:, 0] > 0) & (df_pair.iloc[:, 1] > 0)]
                    if df_pair.shape[0] < 5:
                        corr = np.nan
                    else:
                        corr = df_pair.iloc[:, 0].corr(df_pair.iloc[:, 1])
                    if pd.isna(corr):
                        continue
                    if corr > best_val:
                        best_val = corr
                        best_k = k
                with _topcorr_lock:
                    _topcorr_job_state['progress'] += 1
                    if np.isfinite(best_val):
                        _topcorr_job_state['results'][(s_i, s_j)] = (float(best_val), int(best_k))
                        # Recalculer le top10
                        sorted_pairs = sorted(_topcorr_job_state['results'].items(), key=lambda kv: kv[1][0], reverse=True)
                        top = []
                        for (pi, pj), (val, lag) in sorted_pairs[:10]:
                            top.append((pi, pj, float(val), int(lag)))
                        _topcorr_job_state['top10'] = top
        with _topcorr_lock:
            _topcorr_job_state['running'] = False
    except Exception as e:
        logging.exception("Erreur job top corr")
        with _topcorr_lock:
            _topcorr_job_state['error'] = str(e)
            _topcorr_job_state['running'] = False


def _start_topcorr_job():
    with _topcorr_lock:
        if _topcorr_job_state['running']:
            return False
        _topcorr_job_state['running'] = True
        _topcorr_job_state['progress'] = 0
        _topcorr_job_state['total'] = 0
        _topcorr_job_state['results'] = {}
        _topcorr_job_state['top10'] = []
        _topcorr_job_state['error'] = None
        t = threading.Thread(target=_run_topcorr_job, daemon=True)
        _topcorr_job_state['thread'] = t
        t.start()
        return True

@app.callback(
    [Output('analyse_correlation_heatmap', 'figure'),
     Output('analyse_heatmap_store', 'data')],
    [Input('analyse_shares_dropdown', 'value'),
     Input('analyse_sector_dropdown', 'value'),
     Input('analyse_corr_heatmap_lag', 'value'),
     Input('analyse_heatmap_store', 'data')]
)
def compute_correlation_heatmap(selected_symbols, selected_sector, lag_value, stored_data):
    ctx = dash.callback_context
    if ctx and ctx.triggered and ctx.triggered[0]['prop_id'].startswith('analyse_heatmap_store.data') and stored_data:
        try:
            return go.Figure(stored_data.get('figure', {})), stored_data
        except Exception:
            pass
    # Placeholder figure dark
    placeholder = go.Figure()
    placeholder.update_layout(
        template='plotly_dark',
        title='Sélectionnez au moins deux actions pour afficher la heatmap',
        paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#FFFFFF')
    )

    try:
        symbols = set(selected_symbols or [])
        if selected_sector:
            df = shM.dfShares
            sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
            parts = []
            if 'Non défini' in sel:
                parts.append(df[df['sector'].isna()])
                sel = [s for s in sel if s != 'Non défini']
            if sel:
                parts.append(df[df['sector'].isin(sel)])
            filtered = pd.concat(parts) if parts else pd.DataFrame()
            symbols.update([row.symbol for row in filtered.itertuples()])
        symbols = list(symbols)
        symbols = symbols[:30]

        if len(symbols) < 2:
            return placeholder

        end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=15)

        returns = get_returns_with_daily_fallback(shM, symbols, start_date, end_date)
        if returns.empty or len(returns.columns) < 2:
            return placeholder

        left_cols = [c for c in returns.columns]
        # Forcer la heatmap à utiliser toujours le balayage max 0..30
        lag_value = 'max_0_30'
        if isinstance(lag_value, str) and lag_value == 'max_0_30':
            # Hausses uniquement
            mat, lag_matrix = corr_matrix_max_0_30(returns, positive_mode='both_positive')
            if mat.empty:
                return placeholder
            hover_text = [[f"lag optimal: {int(lag_matrix.loc[i, j])} min" for j in mat.columns] for i in mat.index]
            fig = go.Figure(data=go.Heatmap(
                z=mat.values,
                x=mat.columns,
                y=mat.index,
                colorscale='Viridis', zmin=-1, zmax=1,
                colorbar=dict(title='corr'),
                text=hover_text,
                hovertemplate='%{y} vs %{x}<br>corr=%{z:.3f}<br>%{text}<extra></extra>'
            ))
            fig.update_layout(
                template='plotly_dark',
                title="Heatmap corr (max 0–30 min)",
                paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#FFFFFF')
            )
            payload = {'figure': fig.to_dict()}
            return fig, payload
        elif isinstance(lag_value, (int, np.integer)) and lag_value > 0:
            # Hausses uniquement
            mat = corr_matrix_with_lag(returns, int(lag_value), positive_mode='both_positive')
            if mat.empty:
                return placeholder
        else:
            # Hausses uniquement (lag 0)
            mat = corr_matrix_with_lag(returns, 0, positive_mode='both_positive')

        fig = go.Figure(data=go.Heatmap(
            z=mat.values,
            x=mat.columns,
            y=mat.index,
            colorscale='Viridis', zmin=-1, zmax=1,
            colorbar=dict(title='corr')
        ))
        fig.update_layout(
            template='plotly_dark',
            title="Heatmap corr (max 0–30 min)",
            paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#FFFFFF')
        )
        payload = {'figure': fig.to_dict()}
        return fig, payload
    except Exception as e:
        logging.exception("Erreur heatmap correlation")
        try:
            placeholder.update_layout(title=f"Erreur heatmap: {e}")
        except Exception:
            pass
        return placeholder


CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '20px'
}

INNER_CARD_STYLE = {
    'backgroundColor': '#12121a',
    'padding': '16px',
    'borderRadius': '12px',
    'border': '1px solid rgba(148, 163, 184, 0.1)'
}

SECTION_TITLE_STYLE = {
    'fontSize': '0.8125rem',
    'fontWeight': '500',
    'color': '#94a3b8',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em',
    'marginBottom': '8px'
}

def layout_content():
    help_text = """
### Analyse de Données

Cette page est votre tableau de bord statistique pour comprendre les dynamiques du marché et les relations entre les actions.

#### 1. Filtres
Sélectionnez les actifs à analyser :
*   **Secteurs** : Filtrez par industrie (ex: Tech, Santé).
*   **Actions** : Choisissez des tickers spécifiques.

#### 2. Corrélations (Heatmap)
Une carte thermique montrant quelles actions bougent ensemble.
*   **Pourquoi ?** Pour diversifier votre portefeuille ou trouver des paires de trading.
*   **Couleur** :
    *   Jaune/Vert clair : Forte corrélation positive (elles montent ensemble).
    *   Bleu/Violet : Pas de corrélation ou inverse.
*   **Max (0-30 min)** : Le système cherche automatiquement si une action suit l'autre avec un décalage de 0 à 30 minutes.

#### 3. Corrélation Croisée (Scatter Plot)
Visualisez l'évolution de la corrélation jour après jour entre deux actifs. Cela permet de voir si leur relation est stable ou changeante.

#### 4. Potentiels Quotidiens
Un graphique montrant l'évolution du "score de potentiel" calculé par nos algorithmes pour les actions sélectionnées sur les 15 derniers jours.

#### 5. Top 10 Corrélations
Un outil puissant qui scanne **toutes** les paires d'actions possibles pour vous sortir les 10 meilleures corrélations (avec décalage temporel) du moment. Idéal pour repérer des opportunités de Lead-Lag.
"""

    return html.Div([
        create_page_help("Aide Analyse", help_text),
        
        # En-tête de page
        html.Div([
            html.H3('Analyse', style={
                'margin': '0',
                'textAlign': 'center'
            }),
            html.P('Corrélations & Statistiques de Marché', style={
                'textAlign': 'center',
                'color': '#94a3b8',
                'marginTop': '8px',
                'marginBottom': '0'
            })
        ], style={'marginBottom': '32px'}),
        
        # Stores persistants pour conserver les sorties calculées
        dcc.Store(id='analyse_heatmap_store', storage_type='session'),
        dcc.Store(id='analyse_tables_store', storage_type='session'),

        # Section Filtres
        html.Div([
            html.Div([
                html.Span('🔍', style={'fontSize': '1.25rem'}),
                html.Span('Filtres', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label('Secteur(s)', style=SECTION_TITLE_STYLE),
                    dcc.Dropdown(
                        id='analyse_sector_dropdown',
                        options=[
                            {'label': sector, 'value': sector}
                            for sector in sorted(shM.dfShares['sector'].fillna('Non défini').unique())
                        ],
                        multi=True,
                        placeholder='Sélectionner un secteur...',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ]),
                html.Div([
                    html.Label('Actions', style=SECTION_TITLE_STYLE),
                    dcc.Dropdown(
                        id='analyse_shares_dropdown',
                        options=[
                            {'label': f'{row.symbol}', 'value': row.symbol}
                            for row in shM.dfShares.sort_values(by='symbol').itertuples()
                        ],
                        multi=True,
                        placeholder='Sélectionner des actions...',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    )
                ])
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))',
                'gap': '16px'
            })
        ], style=CARD_STYLE),

        # Section Corrélations
        html.Div([
            html.Div([
                html.Span('📊', style={'fontSize': '1.25rem'}),
                html.Span('Corrélations multi-actifs', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            
            # Tables de corrélation par lag
            html.Div([
                html.Div([
                    html.H6('Décalage 0 min', style={'color': '#f8fafc', 'marginBottom': '8px', 'fontSize': '0.875rem'}),
                    html.Div(id='analyse_corr_table_lag_0')
                ], style=INNER_CARD_STYLE),
                html.Div([
                    html.H6('Décalage 1 min', style={'color': '#f8fafc', 'marginBottom': '8px', 'fontSize': '0.875rem'}),
                    html.Div(id='analyse_corr_table_lag_1')
                ], style=INNER_CARD_STYLE),
                html.Div([
                    html.H6('Décalage 3 min', style={'color': '#f8fafc', 'marginBottom': '8px', 'fontSize': '0.875rem'}),
                    html.Div(id='analyse_corr_table_lag_3')
                ], style=INNER_CARD_STYLE),
                html.Div([
                    html.H6('Décalage 6 min', style={'color': '#f8fafc', 'marginBottom': '8px', 'fontSize': '0.875rem'}),
                    html.Div(id='analyse_corr_table_lag_6')
                ], style=INNER_CARD_STYLE),
                html.Div([
                    html.H6('Décalage 10 min', style={'color': '#f8fafc', 'marginBottom': '8px', 'fontSize': '0.875rem'}),
                    html.Div(id='analyse_corr_table_lag_10')
                ], style=INNER_CARD_STYLE)
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))',
                'gap': '12px',
                'marginBottom': '20px'
            }),
            
            # Heatmap
            html.Div([
                html.H5('🗺️ Heatmap de corrélation', style={'color': '#f8fafc', 'marginBottom': '12px'}),
                dcc.Dropdown(
                    id='analyse_corr_heatmap_lag',
                    options=[
                        {'label': '⏱️ Décalage 0 min', 'value': 0},
                        {'label': '⏱️ Décalage 1 min', 'value': 1},
                        {'label': '⏱️ Décalage 3 min', 'value': 3},
                        {'label': '⏱️ Décalage 6 min', 'value': 6},
                        {'label': '⏱️ Décalage 10 min', 'value': 10},
                        {'label': '🔥 Max (0–30 min)', 'value': 'max_0_30'},
                    ],
                    value='max_0_30',
                    clearable=False,
                    style={'width': '280px', 'marginBottom': '12px'},
                    persistence=True, persistence_type='session'
                ),
                dcc.Graph(id='analyse_correlation_heatmap', style={'height': '45vh'})
            ], style={**INNER_CARD_STYLE, 'marginBottom': '16px'}),
            
            # Cross-correlation scatter
            html.Div([
                html.H5('📈 Corrélation croisée journalière (0–30 min)', style={'color': '#f8fafc', 'marginBottom': '12px'}),
                dcc.Graph(id='analyse_crosscorr_daily_scatter', style={'height': '45vh'})
            ], style=INNER_CARD_STYLE)
        ], style=CARD_STYLE),

        # Section Potentiels quotidiens
        html.Div([
            html.Div([
                html.Span('📉', style={'fontSize': '1.25rem'}),
                html.Span('Potentiels quotidiens — 15 derniers jours', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
            dcc.Loading(dcc.Graph(id='analyse_daily_potential_graph', style={'height': '45vh'}), type='default')
        ], style=CARD_STYLE),

        # Section Statistiques
        html.Div([
            html.Div([
                html.Span('📋', style={'fontSize': '1.25rem'}),
                html.Span('Statistiques par action', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
            
            dcc.Dropdown(
                id='stats_share_list',
                options=[],
                multi=True,
                placeholder='Choisir des actions pour les statistiques...',
                style={'width': '100%', 'marginBottom': '16px'},
                persistence=True, persistence_type='session'
            ),
            html.Div(id='share_stats_panel', style={
                **INNER_CARD_STYLE,
                'color': '#f8fafc'
            })
        ], style=CARD_STYLE),

        # Section Top 10
        html.Div([
            html.Div([
                html.Span('🏆', style={'fontSize': '1.25rem'}),
                html.Span('Top 10 corrélations (0–30 min)', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '16px', 'display': 'flex', 'alignItems': 'center'}),
            
            dcc.Interval(id='analyse_topcorr_poll', interval=1000, n_intervals=0, disabled=True),
            html.Button('🔄 Calculer le Top 10', id='analyse_topcorr_btn', n_clicks=0, style={
                'background': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
                'color': 'white',
                'border': 'none',
                'padding': '12px 24px',
                'borderRadius': '10px',
                'fontWeight': '600',
                'fontFamily': 'Outfit, sans-serif',
                'cursor': 'pointer',
                'boxShadow': '0 4px 15px rgba(99, 102, 241, 0.3)',
                'marginBottom': '16px'
            }),
            html.Div(id='analyse_topcorr_result', style={
                'whiteSpace': 'pre-wrap',
                'color': '#f8fafc',
                'fontFamily': 'JetBrains Mono, monospace',
                'fontSize': '0.875rem',
                'backgroundColor': '#12121a',
                'padding': '16px',
                'borderRadius': '10px',
                'border': '1px solid rgba(148, 163, 184, 0.1)'
            })
        ], style=CARD_STYLE),

        # Spacer pour navigation
        html.Div(style={'height': '100px'}),

        # Navigation
        create_navigation()
    ], style={
        'backgroundColor': '#0a0a0f',
        'minHeight': '100vh',
        'padding': '24px 32px',
        'width': '100%',
        'maxWidth': '100%',
        'margin': '0'
    })


# Pour compatibilité avec index.py qui attend `layout`
layout = layout_content()


# --- Callbacks ---

@app.callback(
    Output('analyse_crosscorr_daily_scatter', 'figure'),
    [Input('analyse_shares_dropdown', 'value'),
     Input('analyse_sector_dropdown', 'value')]
)
def crosscorr_daily_scatter(selected_symbols, selected_sector):
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#FFFFFF'),
                      title='Corrélation croisée journalière (0–30 min) — sélection insuffisante')
    try:
        # Choisir une paire (deux premiers symboles distincts de la sélection/secteur)
        symbols = []
        if selected_symbols:
            symbols.extend(selected_symbols)
        if selected_sector:
            df = shM.dfShares
            sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
            parts = []
            if 'Non défini' in sel:
                parts.append(df[df['sector'].isna()])
                sel = [s for s in sel if s != 'Non défini']
            if sel:
                parts.append(df[df['sector'].isin(sel)])
            filtered = pd.concat(parts) if parts else pd.DataFrame()
            for row in filtered.itertuples():
                if row.symbol not in symbols:
                    symbols.append(row.symbol)
        symbols = [s for s in symbols if s is not None][:2]
        if len(symbols) < 2:
            return fig

        # Fenêtre 15 jours
        end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=15)

        # Points de corrélation croisée par jour via service
        xs, ys, texts = daily_crosscorr_points(shM, symbols[0], symbols[1], start_date, end_date)

        if xs:
            fig = go.Figure(data=go.Scatter(x=xs, y=ys, mode='markers', text=texts,
                                             hovertemplate='%{x|%Y-%m-%d} · %{text}<br>corr=%{y:.3f}<extra></extra>'))
            fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', font=dict(color='#FFFFFF'),
                              title=f"Corrélation croisée journalière (0–30 min) — {symbols[0]} vs {symbols[1]}",
                              xaxis_title='Jour', yaxis_title='Corrélation')
        return fig
    except Exception as e:
        logging.exception("Erreur scatter cross-corr")
        fig.update_layout(title=f"Erreur scatter cross-corr: {e}")
        return fig

@app.callback(
    Output('analyse_shares_dropdown', 'options'),
    Input('analyse_sector_dropdown', 'value')
)
def update_analyse_shares_options(selected_sector):
    df = shM.dfShares
    if selected_sector:
        # multi-sélection: union des secteurs + cas "Non défini"
        sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
        parts = []
        if 'Non défini' in sel:
            parts.append(df[df['sector'].isna()])
            sel = [s for s in sel if s != 'Non défini']
        if sel:
            parts.append(df[df['sector'].isin(sel)])
        filtered_shares = pd.concat(parts) if parts else df
    else:
        filtered_shares = df
    sorted_shares = filtered_shares.sort_values(by='symbol')
    return [
        {'label': f'{row.symbol}', 'value': row.symbol}
        for row in sorted_shares.itertuples()
    ]


## Fonctions de période supprimées


## Callback supprimé: comparaison agrégée par période


@app.callback(
    [Output('analyse_corr_table_lag_0', 'children'),
     Output('analyse_corr_table_lag_1', 'children'),
     Output('analyse_corr_table_lag_3', 'children'),
     Output('analyse_corr_table_lag_6', 'children'),
     Output('analyse_corr_table_lag_10', 'children'),
     Output('analyse_tables_store', 'data')],
    [Input('analyse_shares_dropdown', 'value'),
     Input('analyse_sector_dropdown', 'value'),
     Input('analyse_tables_store', 'data')]
)
def compute_correlation_tables(selected_symbols, selected_sector, stored_tables):
    ctx = dash.callback_context
    if ctx and ctx.triggered and ctx.triggered[0]['prop_id'].startswith('analyse_tables_store.data') and stored_tables:
        try:
            # Réhydrater les 5 tables
            def make_table(records):
                if not records:
                    return html.Em("Corrélation non disponible")
                columns = [{"name": c, "id": c} for c in records[0].keys()]
                return dash_table.DataTable(
                    data=records,
                    columns=columns,
                    style_table={"overflowX": "auto", "maxHeight": "40vh", "overflowY": "auto"},
                    style_cell={"textAlign": "center", "minWidth": "80px", "width": "80px", "maxWidth": "120px"},
                    style_header={"backgroundColor": "#000000", "color": "#ffffff"},
                    style_data={"backgroundColor": "#1E1E1E", "color": "#ffffff"},
                )
            t0 = make_table(stored_tables.get('lag0'))
            t1 = make_table(stored_tables.get('lag1'))
            t3 = make_table(stored_tables.get('lag3'))
            t6 = make_table(stored_tables.get('lag6'))
            t10 = make_table(stored_tables.get('lag10'))
            return t0, t1, t3, t6, t10, stored_tables
        except Exception:
            pass
    symbols = set(selected_symbols or [])
    if selected_sector:
        df = shM.dfShares
        sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
        parts = []
        if 'Non défini' in sel:
            parts.append(df[df['sector'].isna()])
            sel = [s for s in sel if s != 'Non défini']
        if sel:
            parts.append(df[df['sector'].isin(sel)])
        filtered = pd.concat(parts) if parts else pd.DataFrame()
        symbols.update([row.symbol for row in filtered.itertuples()])
    symbols = list(symbols)
    # Pas de valeurs par défaut: on n'affiche rien si <2
    symbols = symbols[:25]
    if len(symbols) < 2:
        msg = html.Em("Sélectionnez au moins deux actions pour afficher la corrélation.")
        return msg, msg, msg, msg, msg, None

    # Fenêtre des 15 derniers jours (calendaires) à partir d'aujourd'hui inclus
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=15)

    # Utiliser le service: rendements minute (main hours + exclusion paires incompatibles)
    returns = get_minute_returns(shM, symbols, start_date, end_date)
    if returns.empty or len(returns.columns) < 2:
        msg = html.Em("Données insuffisantes pour calculer les corrélations.")
        return msg, msg, msg, msg, msg, None

    mats = {}
    for lag in [0, 1, 3, 6, 10]:
        # Hausses uniquement, version rapide
        mat = corr_matrix_with_lag_fast(returns, lag, positive_mode='both_positive')
        if mat is None or mat.empty:
            mats[lag] = html.Em("Corrélation non disponible")
        else:
            mats[lag] = dash_table.DataTable(
                data=mat.round(3).reset_index().rename(columns={'index': 'Symbol'}).to_dict('records'),
                columns=[{"name": c, "id": c} for c in ['Symbol'] + [c for c in mat.columns]],
                style_table={"overflowX": "auto", "maxHeight": "40vh", "overflowY": "auto"},
                style_cell={"textAlign": "center", "minWidth": "80px", "width": "80px", "maxWidth": "120px"},
                style_header={"backgroundColor": "#000000", "color": "#ffffff"},
                style_data={"backgroundColor": "#1E1E1E", "color": "#ffffff"},
            )

    # Sauvegarder au format records pour réhydratation
    def to_records(mat_or_component):
        if isinstance(mat_or_component, dash_table.DataTable):
            return mat_or_component.data
        return None
    stored_payload = {
        'lag0': to_records(mats[0]),
        'lag1': to_records(mats[1]),
        'lag3': to_records(mats[3]),
        'lag6': to_records(mats[6]),
        'lag10': to_records(mats[10])
    }
    return mats[0], mats[1], mats[3], mats[6], mats[10], stored_payload


@app.callback(
    [Output('stats_share_list', 'options'), Output('stats_share_list', 'value')],
    [Input('analyse_shares_dropdown', 'value'), Input('analyse_sector_dropdown', 'value')]
)
def populate_stats_list_from_analyse(selected_symbols, selected_sector):
    symbols = set(selected_symbols or [])
    if selected_sector:
        df = shM.dfShares
        sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
        parts = []
        if 'Non défini' in sel:
            parts.append(df[df['sector'].isna()])
            sel = [s for s in sel if s != 'Non défini']
        if sel:
            parts.append(df[df['sector'].isin(sel)])
        filtered = pd.concat(parts) if parts else pd.DataFrame()
        symbols.update([row.symbol for row in filtered.itertuples()])
    symbols = list(symbols)
    # Valeurs par défaut uniquement pour ce composant Stats
    default_syms = symbols[:5] if symbols else []
    opts = [{'label': s, 'value': s} for s in (symbols if symbols else [])]
    return opts, default_syms


@app.callback(
    Output('analyse_daily_potential_graph', 'figure'),
    [Input('analyse_shares_dropdown', 'value'), Input('analyse_sector_dropdown', 'value')]
)
def plot_daily_potential(selected_symbols, selected_sector):
    try:
        symbols = set(selected_symbols or [])
        if selected_sector:
            df = shM.dfShares
            sel = selected_sector if isinstance(selected_sector, list) else [selected_sector]
            parts = []
            if 'Non défini' in sel:
                parts.append(df[df['sector'].isna()])
                sel = [s for s in sel if s != 'Non défini']
            if sel:
                parts.append(df[df['sector'].isin(sel)])
            filtered = pd.concat(parts) if parts else pd.DataFrame()
            symbols.update([row.symbol for row in filtered.itertuples()])
        symbols = list(symbols)
        # Pas de valeurs par défaut ici non plus: on affiche un placeholder
        symbols = symbols[:10]
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            title="Potentiel moyen quotidien (sélection par défaut si non spécifié)",
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        if not symbols:
            fig.update_layout(title="Sélectionnez des actions pour afficher le potentiel quotidien")
            return fig

        today = pd.Timestamp.today().normalize()
        end_N = today
        start_N = end_N - pd.Timedelta(days=14)
        start_N_1 = start_N.replace(year=start_N.year - 1)

        potential_levels = [30, 15, 10, 7, 5, 4, 3, 2, 1]

        def read_daily_mean(shareObj, start, end):
            dfP = shM.readPotentialsPercentTotal(shareObj, start, end, potential_levels=potential_levels)
            if dfP is None or dfP.empty:
                return pd.Series(dtype=float)
            cols = [f"{lvl}_percentTotal" for lvl in potential_levels if f"{lvl}_percentTotal" in dfP.columns]
            if not cols:
                return pd.Series(dtype=float)
            # S'assurer d'avoir une colonne date ou index utilisable
            if 'date' in dfP.columns:
                idx = pd.to_datetime(dfP['date'])
            elif 'time' in dfP.columns:
                idx = pd.to_datetime(dfP['time'])
            else:
                idx = pd.RangeIndex(len(dfP))
            # Convertir en numérique colonne par colonne pour éviter l'erreur d'arg
            df_num = dfP[cols].apply(pd.to_numeric, errors='coerce')
            daily_mean = df_num.mean(axis=1)
            daily_mean.index = idx
            # Ramener à une moyenne par jour si plusieurs points intrajournaliers
            return daily_mean.resample('1D').mean()

        for symbol in symbols:
            dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
            if dfShares.empty:
                continue
            shareObj = dfShares.iloc[0]
            # Mise à jour des potentiels sur la fenêtre si manquants
            try:
                # Mettre à jour les cotations avant le calcul des potentiels
                shM.updateShareCotations(shareObj, checkDuplicate=False)
                all_days = pd.date_range(start=start_N, end=end_N, freq='D').date
                shM.computePotentialForDates(shareObj, list(all_days), potential_levels=potential_levels)
            except Exception as e:
                logging.warning(f"Impossible de mettre à jour les potentiels pour {symbol}: {e}")
            sN = read_daily_mean(shareObj, start_N, end_N)
            if not sN.empty:
                fig.add_trace(go.Scatter(x=sN.index, y=sN.values, mode='lines+markers', name=f"{symbol}"))

        fig.update_layout(title='Potentiel moyen quotidien (moyenne sur k) — 15 derniers jours', xaxis_title='Date', yaxis_title='PercentTotal moyen')
        return fig
    except Exception as e:
        logging.exception("Erreur potentiel quotidien")
        fig = go.Figure()
        fig.update_layout(
            template='plotly_dark',
            title=f"Erreur potentiel quotidien: {e}",
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font=dict(color='#FFFFFF')
        )
        return fig


@app.callback(
    [Output('analyse_topcorr_result', 'children'),
     Output('analyse_topcorr_poll', 'disabled')],
    [Input('analyse_topcorr_btn', 'n_clicks'),
     Input('analyse_topcorr_poll', 'n_intervals')]
)
def compute_top10_global_correlations(n_clicks, n_intervals):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else ''
    # Si clic bouton => démarrer le job si pas déjà en cours et activer le polling
    if triggered.startswith('analyse_topcorr_btn'):
        started = _start_topcorr_job()
        # Activer polling (disabled=False)
        # Rendu initial: message de démarrage
        return html.Em('Calcul en cours…'), False
    # Si tick du polling => lire l'état et afficher
    with _topcorr_lock:
        running = _topcorr_job_state['running']
        progress = _topcorr_job_state['progress']
        total = _topcorr_job_state['total']
        error = _topcorr_job_state['error']
        top10 = list(_topcorr_job_state['top10'])
    if error:
        return html.Em(error), True
    if total > 0:
        header = f"Progression: {progress}/{total} paires traitées"
    else:
        header = "Initialisation…"
    if top10:
        lines = [header]
        for rank, (i, j, val, lag) in enumerate(top10, start=1):
            lines.append(f"{rank}. {i} ↔ {j} | lag={lag} min | corr={val:.3f}")
        content = html.Pre("\n".join(lines), style={'whiteSpace': 'pre-wrap', 'color': '#FFFFFF'})
    else:
        content = html.Pre(header, style={'whiteSpace': 'pre-wrap', 'color': '#FFFFFF'})
    # Désactiver le polling si le job a fini
    return content, (False if running else True)
