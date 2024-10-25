#prediction.py
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objs as go

import datetime

from web.app import app, shM
from web.apps.sharedFig import fig  # Remplacement de l'importation relative par une importation absolue
from Models import lstm as ls  # Remplacement de l'importation relative par une importation absolue

layout = html.Div([
    html.H3('Prediction'),
    dcc.Dropdown(
        id='dashboard_dropdown',
        options=[
            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} for stock in shM.dfShares.itertuples()
        ],
        multi=True
    ),
    html.Div([
        dcc.Graph(id='stock_graph', figure=fig)
    ], style={'width': '100%'}),  # Le graphique occupe toute la largeur

    # Déplacer le bouton Normalize et les liens sous le graphique
    html.Div([
        daq.BooleanSwitch(id='boolean_switch_normalize', label="Normalize", on=False, color="#f80"),
        html.Br(),
        dcc.Link('Go update', href='/update'),
        html.Br(),
        dcc.Link('Go to dashboard', href='/dashboard')
    ], style={'width': '100%', 'textAlign': 'center'}),  # Centrer les éléments
])

@app.callback(
    Output('stock_graph', 'figure'),  # Correction de l'ID de sortie
    Input('dashboard_dropdown', 'value'),
    Input('boolean_switch_normalize', 'on')
    )
def display_prediction_stock_graph(values, toNormalize):
    # Ajout de traces pour vérifier les données
    print("Selected values:", values)
    print("Normalize:", toNormalize)

    if values is None:
        print("No values selected.")
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig

    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(hovermode="x unified")
    fig.update_layout(hoverlabel_bgcolor='#00FF00')
    fig.update_layout(hoverlabel_bordercolor='#00FF00')
    fig.update_layout(legend_bgcolor='#00FF00')
    fig.layout.title = 'Shares quots'
    
    # Vérification des données récupérées
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values), values, op='|')
    print("DataFrame for selected values:", dfShares)

    listDfData = []  # Définition de la liste

    for curShare in dfShares.itertuples():
        data_quots = shM.get_cotations_data_df(curShare, curShare.firstRecord, curShare.lastRecord)
        print(f"Data for {curShare.symbol}:", data_quots)
        testPredict, testX = ls.compute_lstm(curShare, data_quots)
        listDfData.append(data_quots)  # Ajout des données à la liste

    for i, dfData in enumerate(listDfData):
        if not dfData.empty:
            print(f"Data for {values[i]}:", dfData)
            if toNormalize and dfData['openPrice'][0] > 0:
                dfData['openPrice'] /= dfData['openPrice'][0]
            fig.add_scatter(name=values[i], x=dfData.index.values, y=dfData['openPrice'].values)
        else:
            print(f"No data for {values[i]}")

    return fig