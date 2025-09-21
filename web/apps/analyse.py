from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from app import app, shM
from web.components.navigation import create_navigation
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def layout_content():
    return html.Div([
        html.H3('Analyse', style={'color': '#FF8C00'}),

        # Filtres
        html.Div([
            html.Div([
                html.Label('Secteur(s)'),
                dcc.Dropdown(
                    id='analyse_sector_dropdown',
                    options=[
                        {'label': sector, 'value': sector}
                        for sector in sorted(shM.dfShares['sector'].fillna('Non défini').unique())
                    ],
                    multi=True,
                    placeholder='Sélectionner un secteur',
                    style={'width': '100%', 'color': '#FF8C00'},
                    persistence=True, persistence_type='memory'
                )
            ]),
            html.Div([
                html.Label('Actions'),
                dcc.Dropdown(
                    id='analyse_shares_dropdown',
                    options=[
                        {'label': f'{row.symbol}', 'value': row.symbol}
                        for row in shM.dfShares.sort_values(by='symbol').itertuples()
                    ],
                    multi=True,
                    placeholder='Sélectionner des actions',
                    style={'width': '100%', 'color': '#FF8C00'},
                    persistence=True, persistence_type='memory'
                )
            ])
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(260px, 1fr))', 'gap': '10px', 'backgroundColor': '#2E2E2E', 'padding': '10px', 'borderRadius': '8px'}),

        # Périodes
        html.Div([
            html.Label('Périodes prédéfinies'),
            dcc.Checklist(
                id='analyse_periods_checklist',
                options=[
                    {'label': 'Année en cours (N)', 'value': 'year_N'},
                    {'label': 'Année précédente (N-1)', 'value': 'year_N1'},
                    {'label': 'Derniers 15 jours ouvrés (N)', 'value': '15dN'},
                    {'label': 'Fenêtre correspondante en N-1', 'value': '15dN1'},
                ],
                value=['year_N', 'year_N1', '15dN', '15dN1'],
                inline=True,
                style={'color': '#FF8C00'}
            )
        ], style={'backgroundColor': '#2E2E2E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '10px'}),

        # Tableaux et graphiques
        html.Div([
            html.Div([
                html.H5('Comparaison agrégée par période (graphe)'),
                dcc.Loading(html.Div(id='analyse_aggregate_table'), type='default')
            ], style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px'}),

            html.Div([
                html.H5('Corrélation multi-actifs (openPrice)'),
                dcc.Graph(id='analyse_correlation_heatmap', style={'height': '45vh'})
            ], style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px'})
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '10px', 'marginTop': '10px'}),

        # Graphe potentiels quotidiens — 15 derniers jours (une courbe par action)
        html.Div([
            html.H5('Potentiels quotidiens — 15 derniers jours ouvrés'),
            dcc.Loading(dcc.Graph(id='analyse_daily_potential_graph', style={'height': '45vh'}), type='default')
        ], style={'backgroundColor': '#1E1E1E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '10px'}),

        # Statistiques par action (déplacé depuis Prédiction)
        html.Div([
            html.Div('Statistiques par action', style={'color': '#FF8C00', 'marginBottom': '8px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='stats_share_list',
                options=[],
                multi=True,
                placeholder='Choisir des actions pour les statistiques',
                style={'width': '100%', 'color': '#FF8C00'}
            ),
            html.Div(id='share_stats_panel', style={
                'marginTop': '10px',
                'backgroundColor': '#1E1E1E',
                'padding': '10px',
                'borderRadius': '8px',
                'color': '#FFFFFF'
            })
        ], style={'backgroundColor': '#2E2E2E', 'padding': '10px', 'borderRadius': '8px', 'marginTop': '10px'}),

        # Navigation
        create_navigation()
    ], style={'backgroundColor': 'black', 'minHeight': '100vh', 'padding': '20px'})


# Pour compatibilité avec index.py qui attend `layout`
layout = layout_content()


# --- Callbacks ---

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


def _periods_from_keys(keys):
    now = pd.Timestamp.today().normalize()
    # N / N-1 et fenêtre 15 jours ouvrés
    start_year_N = pd.Timestamp(year=now.year, month=1, day=1)
    end_year_N = now
    start_year_N1 = pd.Timestamp(year=now.year - 1, month=1, day=1)
    end_year_N1 = pd.Timestamp(year=now.year - 1, month=12, day=31, hour=23, minute=59, second=59)
    days_N = pd.bdate_range(end=now, periods=15)
    days_N1 = days_N - pd.DateOffset(years=1)
    mapping = {
        'year_N': ('Année N', start_year_N, end_year_N),
        'year_N1': ('Année N-1', start_year_N1, end_year_N1),
        '15dN': ('15 derniers jours ouvrés (N)', days_N.min(), days_N.max()),
        '15dN1': ('Fenêtre N-1 (15 jours ouvrés calendaire)', days_N1.min(), days_N1.max()),
    }
    results = []
    for k in keys or []:
        if k in mapping:
            results.append(mapping[k])
    return results


@app.callback(
    Output('analyse_aggregate_table', 'children'),
    [Input('analyse_shares_dropdown', 'value'),
     Input('analyse_sector_dropdown', 'value'),
     Input('analyse_periods_checklist', 'value')]
)
def compute_aggregate_table(selected_symbols, selected_sector, selected_period_keys):
    # Union: actions choisies + secteurs sélectionnés
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
    if not symbols:
        return html.Em("Sélectionnez un secteur ou des actions pour afficher le graphe.")

    periods_meta = _periods_from_keys(selected_period_keys)
    if not periods_meta:
        return html.Em("Sélectionnez au moins une période.")

    potential_levels = [1, 2, 3, 4, 5, 7, 10, 15, 30]
    # Accumulateur: période -> { 'count': n, 'k=1': [vals...], ... }
    agg = {}
    for label, start, end in periods_meta:
        agg[label] = {'Nb actions': 0}
        for lvl in potential_levels:
            agg[label][f'k={lvl}'] = []

    for symbol in symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if dfShares.empty:
            continue
        shareObj = dfShares.iloc[0]
        for label, start, end in periods_meta:
            dfP = shM.readPotentialsPercentTotal(shareObj, start, end, potential_levels=potential_levels)
            if dfP is not None and not dfP.empty:
                agg[label]['Nb actions'] += 1
                for lvl in potential_levels:
                    col = f'{lvl}_percentTotal'
                    if col in dfP.columns:
                        vals = pd.to_numeric(dfP[col], errors='coerce').dropna()
                        if not vals.empty:
                            agg[label][f'k={lvl}'].append(float(vals.mean()))

    # Construire un graphe: barres par période, une barre par k (ou moyenne sur k)
    # Ici: on affiche la moyenne sur k= [1,2,3,4,5,7,10,15,30]
    import plotly.graph_objects as go
    fig = go.Figure()
    for label, _, _ in periods_meta:
        vals = []
        for lvl in potential_levels:
            vlist = agg[label][f'k={lvl}']
            v = (sum(vlist)/len(vlist)) if vlist else None
            if v is not None:
                vals.append(v)
        mean_val = (sum(vals)/len(vals)) if vals else None
        fig.add_bar(name=label, x=[label], y=[mean_val or 0])
    fig.update_layout(
        template='plotly_dark',
        title='PercentTotal moyen par période (moyenne sur k)',
        yaxis_title='PercentTotal moyen',
        xaxis_title='Périodes',
        barmode='group'
    )
    return dcc.Graph(figure=fig)


@app.callback(
    Output('analyse_correlation_heatmap', 'figure'),
    [Input('analyse_shares_dropdown', 'value'),
     Input('analyse_sector_dropdown', 'value'),
     Input('analyse_periods_checklist', 'value')]
)
def compute_correlation_heatmap(selected_symbols, selected_sector, selected_period_keys):
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
    # Limiter pour performance
    symbols = symbols[:30]
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', title='Corrélation indisponible (sélection insuffisante)')
    if len(symbols) < 2:
        return fig

    # Choisir une période (priorité à 1m, sinon 6m, 18m, 1w)
    pref = ['1m', '6m', '18m', '1w']
    chosen = next((k for k in pref if k in (selected_period_keys or [])), '1m')
    period_list = _periods_from_keys([chosen])
    if not period_list:
        period_list = _periods_from_keys(['1m'])
    label, start, end = period_list[0]
    # Récupérer données
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(symbols), symbols, op='|')
    listDf = shM.getListDfDataFromDfShares(dfShares, start, end)
    # Construire un DataFrame prix aligné
    price_map = {}
    for i, df in enumerate(listDf):
        if df is None or df.empty:
            continue
        sym = symbols[i] if i < len(symbols) else f's{i}'
        s = pd.to_numeric(df.get('openPrice', pd.Series(dtype=float)), errors='coerce')
        series = pd.Series(s.values, index=pd.to_datetime(df.index))
        # Ramener à la fréquence quotidienne (première cotation)
        daily = series.resample('1D').first().dropna()
        price_map[sym] = daily
    if len(price_map) < 2:
        return fig
    aligned = pd.DataFrame(price_map).dropna(how='any')
    if aligned.shape[0] < 3:
        return fig
    returns = aligned.pct_change().dropna(how='any')
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='Viridis',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='corr')
    ))
    fig.update_layout(template='plotly_dark', title='Corrélation des rendements (quotidienne)')
    return fig


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
    opts = [{'label': s, 'value': s} for s in symbols]
    return opts, symbols[:5]


@app.callback(
    Output('analyse_daily_potential_graph', 'figure'),
    [Input('analyse_shares_dropdown', 'value'), Input('analyse_sector_dropdown', 'value')]
)
def plot_daily_potential(selected_symbols, selected_sector):
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
    symbols = symbols[:10]
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', title="Sélectionnez au moins 1 action")
    if not symbols:
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
        daily_mean = pd.to_numeric(dfP[cols], errors='coerce').mean(axis=1)
        daily_mean.index = idx
        # Ramener à une moyenne par jour si plusieurs points intrajournaliers
        return daily_mean.resample('1D').mean()

    for symbol in symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if dfShares.empty:
            continue
        shareObj = dfShares.iloc[0]
        sN = read_daily_mean(shareObj, start_N, end_N)
        if not sN.empty:
            fig.add_trace(go.Scatter(x=sN.index, y=sN.values, mode='lines+markers', name=f"{symbol}"))

    fig.update_layout(title='Potentiel moyen quotidien (moyenne sur k) — 15 derniers jours', xaxis_title='Date', yaxis_title='PercentTotal moyen')
    return fig

