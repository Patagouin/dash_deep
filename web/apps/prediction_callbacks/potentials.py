from dash.dependencies import Input, Output, ALL
from dash import html, dash_table, dcc
import plotly.graph_objects as go
from app import app, shM
import pandas as pd
import logging


@app.callback(
    Output({'type': 'potential-section', 'symbol': ALL}, 'children'),
    Input('stats_share_list', 'value')
)
def display_potential_tables(selected_symbols):
    if not selected_symbols:
        return []

    outputs = []
    today = pd.Timestamp.today().normalize()

    # Liste des k demandés
    potential_levels = [1, 2, 3, 4, 5, 7, 10, 15, 30]

    for symbol in selected_symbols:
        dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
        if dfShares.empty:
            outputs.append(html.Div([html.Strong(f"Aucune donnée trouvée pour {symbol}.")]))
            continue
        shareObj = dfShares.iloc[0]

        # Définir: 15 derniers jours calendaires
        start_N = today - pd.Timedelta(days=14)
        end_N = today

        # Mise à jour des potentiels pour ces jours
        try:
            all_days = pd.date_range(start=start_N, end=end_N, freq='D').date
            shM.computePotentialForDates(shareObj, list(all_days), potential_levels=[30, 15, 10, 7, 5, 4, 3, 2, 1])
        except Exception as e:
            logging.warning(f"Impossible de mettre à jour les potentiels pour {symbol}: {e}")

        # Helper: moyenne quotidienne sur l'ensemble des k (percentTotal)
        def read_daily_mean(start, end):
            dfP = shM.readPotentialsPercentTotal(shareObj, start, end, potential_levels=potential_levels)
            if dfP is None or dfP.empty:
                return pd.Series(dtype=float)
            cols = [f"{lvl}_percentTotal" for lvl in potential_levels if f"{lvl}_percentTotal" in dfP.columns]
            if not cols:
                return pd.Series(dtype=float)
            if 'date' in dfP.columns:
                idx = pd.to_datetime(dfP['date'])
            elif 'time' in dfP.columns:
                idx = pd.to_datetime(dfP['time'])
            else:
                idx = pd.RangeIndex(len(dfP))
            s = pd.to_numeric(dfP[cols], errors='coerce').mean(axis=1)
            s.index = idx
            # Ramener à une moyenne par jour (si plusieurs enregistrements)
            return s.resample('1D').mean()

        series_N = read_daily_mean(start_N, end_N)

        # Construire le graphe
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', title=f"Potentiel quotidien — 15 derniers jours — {symbol}")
        if not series_N.empty:
            fig.add_trace(go.Scatter(x=series_N.index, y=series_N.values, mode='lines+markers', name=symbol))
        fig.update_layout(xaxis_title='Date', yaxis_title='PercentTotal moyen')

        outputs.append(html.Div([
            html.H6(f"Potentiel quotidien — 15 derniers jours — {symbol}", style={'color': '#4CAF50'}),
            dcc.Loading(dcc.Graph(figure=fig), type='default')
        ]))

    return outputs


