from dash.dependencies import Input, Output
from dash import html, dash_table, dcc
from app import app, shM
import pandas as pd
import logging
import traceback


"""
Affichage des statistiques pour plusieurs actions.
Pour chaque action sélectionnée à droite, on met à jour les cotations,
puis on calcule un tableau des statistiques pour les fenêtres temporelles:
 - Année N-3 (année civile)
 - Année N-1 (année civile)
 - 6 mois avant, 3 mois avant, 1 mois avant, 1 semaine avant, veille
"""
@app.callback(
    Output('share_stats_panel', 'children'),
    Input('stats_share_list', 'value')
)
def display_share_statistics(selected_symbols):
    if not selected_symbols:
        return html.Div([
            html.Em("Sélectionnez une ou plusieurs actions pour afficher leurs statistiques.")
        ])

    children = []

    for selected_symbol in selected_symbols:
        try:
            dfShares = shM.getRowsDfByKeysValues(['symbol'], [selected_symbol])
            if dfShares.empty:
                children.append(html.Div([html.Strong(f"Aucune donnée trouvée pour {selected_symbol}.")]))
                continue

            shareObj = dfShares.iloc[0]

            # Mettre à jour les cotations avant calcul
            try:
                shM.updateShareCotations(shareObj, checkDuplicate=False)
            except Exception as upd_err:
                logging.warning(f"Mise à jour échouée pour {selected_symbol}: {upd_err}")

            # Récupérer les données à jour
            listDfData = shM.getListDfDataFromDfShares(dfShares)
            if not listDfData or len(listDfData) == 0 or listDfData[0].empty:
                children.append(html.Div([html.Strong(f"Aucune cotation disponible pour {selected_symbol}.")]))
                continue

            dfData = listDfData[0].copy()
            dfData.index = pd.to_datetime(dfData.index)

            # Déterminer la date de référence (dernière cotation)
            ref_date = dfData.index.max()

            # Définir les fenêtres temporelles
            def period_year(y):
                start = pd.Timestamp(year=y, month=1, day=1)
                end = pd.Timestamp(year=y, month=12, day=31, hour=23, minute=59, second=59)
                return start, end

            periods = []
            periods.append(("18 derniers mois", ref_date - pd.DateOffset(months=18), ref_date))
            periods.append(("6 derniers mois", ref_date - pd.DateOffset(months=6), ref_date))
            periods.append(("Dernier mois", ref_date - pd.DateOffset(months=1), ref_date))
            periods.append(("Dernière semaine", ref_date - pd.Timedelta(days=7), ref_date))

            # Calcul des métriques par période
            table_rows = []
            for label, start, end in periods:
                dfw = dfData[(dfData.index >= start) & (dfData.index <= end)]
                nb_cotations = int(dfw.shape[0])

                if nb_cotations > 0:
                    first_date = dfw.index.min().strftime('%Y-%m-%d %H:%M')
                    last_date = dfw.index.max().strftime('%Y-%m-%d %H:%M')
                else:
                    first_date = 'N/A'
                    last_date = 'N/A'

                # Ecart-type journalier (avec repli intraday pour "La veille")
                daily_std_str = 'N/A'
                if 'openPrice' in dfw.columns:
                    try:
                        series = pd.Series(dfw['openPrice'].values, index=dfw.index).astype(float)
                        if nb_cotations > 1:
                            daily_price = series.resample('1D').first().dropna()
                            daily_returns = daily_price.pct_change().dropna()
                            if len(daily_returns) > 1:
                                daily_std = float(daily_returns.std())
                                daily_std_str = f"{daily_std*100:.2f}%"

                        # Repli intraday pour la période "La veille" si le calcul précédent n'a rien donné
                        if daily_std_str == 'N/A' and label == 'La veille':
                            intra_returns = series.pct_change().dropna()
                            if len(intra_returns) > 1:
                                intra_std = float(intra_returns.std())
                                daily_std_str = f"{intra_std*100:.2f}%"
                    except Exception as e:
                        logging.warning(f"Unable to compute std for {selected_symbol} ({label}): {e}")

                # Dividendes
                dividends_str = 'Aucun'
                if 'dividend' in dfw.columns:
                    try:
                        div_df = dfw[dfw['dividend'].fillna(0) > 0]
                        if not div_df.empty:
                            dates = sorted(set(pd.to_datetime(div_df.index).date))
                            dividends_str = ', '.join([d.strftime('%Y-%m-%d') for d in dates])
                    except Exception as e:
                        logging.warning(f"Unable to extract dividends for {selected_symbol} ({label}): {e}")

                table_rows.append({
                    'Période': label,
                    'Nb cotations': nb_cotations,
                    'Première cotation': first_date,
                    'Dernière cotation': last_date,
                    'Écart-type journalier': daily_std_str,
                    'Dividendes (dates)': dividends_str
                })

            columns = [
                {'name': 'Période', 'id': 'Période'},
                {'name': 'Nb cotations', 'id': 'Nb cotations'},
                {'name': 'Première cotation', 'id': 'Première cotation'},
                {'name': 'Dernière cotation', 'id': 'Dernière cotation'},
                {'name': 'Écart-type journalier', 'id': 'Écart-type journalier'},
                {'name': 'Dividendes (dates)', 'id': 'Dividendes (dates)'}
            ]

            table = dash_table.DataTable(
                columns=columns,
                data=table_rows,
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
            )

            children.append(html.Div([
                html.H5(f"Statistiques pour {selected_symbol}", style={'color': '#FF8C00'}),
                dcc.Loading(table, type='default'),
                dcc.Loading(
                    html.Div(
                        id={'type': 'potential-section', 'symbol': selected_symbol},
                        children=html.Div("Chargement des potentiels...", style={'color': '#AAAAAA'})
                    ),
                    type='default'
                )
            ], style={'marginBottom': '20px'}))

        except Exception as e:
            logging.error(f"Error while computing statistics for {selected_symbol}: {e}")
            children.append(html.Div([
                html.H5(f"Erreur lors du calcul des statistiques pour {selected_symbol}"),
                html.Pre(str(e)),
                html.Pre(traceback.format_exc())
            ]))

    return children


