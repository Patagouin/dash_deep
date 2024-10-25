#prediction.py
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
import datetime
import numpy as np

from web.app import app, shM
from Models import lstm as ls

layout = html.Div([
    html.H3('Stock Analysis & Prediction'),
    
    # Dropdown for selecting stocks
    dcc.Dropdown(
        id='prediction_dropdown',
        options=[
            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
            for stock in sorted(shM.dfShares.itertuples(), key=lambda stock: stock.symbol)
        ],
        multi=True,
        style={'width': '50%', 'margin': '10px auto'}
    ),

    # Graph container
    html.Div([
        dcc.Graph(
            id='prediction_graph',
            config={'scrollZoom': True},
            style={
                'backgroundColor': 'black',
                'height': '50vh'  # Réduit de 70vh à 50vh
            }
        )
    ], style={
        'width': '100%',
        'margin': '20px 0',
        'backgroundColor': 'black',
        'padding': '20px 0'
    }),

    # Controls section sous le graphique
    html.Div([
        # Date range picker et RadioItems dans le même conteneur
        html.Div([
            dcc.DatePickerRange(
                id='date_picker_range',
                display_format='DD/MM/YY',
                start_date=datetime.datetime.now()-datetime.timedelta(days=7),
                end_date=datetime.datetime.now()
            ),
            html.Div([
                dcc.RadioItems(
                    id='data_type_selector',
                    options=[
                        {'label': 'All Data', 'value': 'all'},
                        {'label': 'Main Hours', 'value': 'main'}
                    ],
                    value='all',
                    labelStyle={'display': 'inline-block', 'margin': '0 10px'}
                )
            ], style={'display': 'inline-block', 'marginLeft': '20px'})
        ], style={'margin': '20px 0'}),

        # Normalize switch séparé
        html.Div([
            daq.BooleanSwitch(
                id='normalize_switch',
                label="Normalize Data",
                on=False,
                color="#4CAF50",
                labelPosition="top"
            )
        ], style={'margin': '20px 0'}),

        # Generate Prediction button
        html.Div([
            html.Button(
                'Generate Prediction',
                id='predict_button',
                n_clicks=0,
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'padding': '10px 20px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '16px'
                }
            )
        ], style={'margin': '20px 0'}),

        # Navigation standardisée avec séparateur
        html.Div([
            html.Hr(style={
                'width': '50%',
                'margin': '20px auto',
                'borderTop': '1px solid #666'
            }),
            html.Div([
                dcc.Link('Dashboard', href='/dashboard', style={'color': '#4CAF50', 'textDecoration': 'none'}),
                html.Span(' | ', style={'margin': '0 10px', 'color': '#666'}),
                dcc.Link('Update', href='/update', style={'color': '#4CAF50', 'textDecoration': 'none'}),
                html.Span(' | ', style={'margin': '0 10px', 'color': '#666'}),
                dcc.Link('Config', href='/config', style={'color': '#4CAF50', 'textDecoration': 'none'})
            ], style={'textAlign': 'center'})
        ])
    ], style={
        'width': '100%',
        'textAlign': 'center',
        'backgroundColor': 'black',
        'padding': '20px 0',
        'color': 'white'  # Pour le texte des contrôles
    }),
], style={'backgroundColor': 'black', 'minHeight': '100vh'})  # Fond noir pour toute la page

@app.callback(
    Output('prediction_graph', 'figure'),
    [Input('prediction_dropdown', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('normalize_switch', 'on'),
     Input('predict_button', 'n_clicks'),
     Input('data_type_selector', 'value')],
    [State('prediction_graph', 'relayoutData')]
)
def update_graph(values, start_date, end_date, normalize, n_clicks, data_type, relayoutData):
    if values is None or len(values) == 0:
        fig = go.Figure()
        fig.layout.template = 'plotly_dark'
        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            font={'color': 'white'},
            margin=dict(t=0)  # Supprime la marge supérieure
        )
        return fig
    
    # Conversion des dates
    if end_date is not None:
        dateLast = datetime.datetime.fromisoformat(end_date)
    else:
        dateLast = datetime.datetime.now()

    if start_date is not None:
        dateBegin = datetime.datetime.fromisoformat(start_date)
    else:
        dateBegin = dateLast - datetime.timedelta(days=7)

    # Création de la figure avec le style dark
    fig = go.Figure()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        font={'color': 'white'},
        hovermode="x unified",
        hoverlabel_bgcolor='#00FF00',
        hoverlabel_bordercolor='#00FF00',
        legend_bgcolor='rgba(0,0,0,0.5)',
        margin=dict(t=0),  # Supprime la marge supérieure
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.2)'
        )
    )
    
    # Récupération et affichage des données
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values), values, op='|')
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)

    for i, (dfData, share) in enumerate(zip(listDfData, dfShares.itertuples())):
        if not dfData.empty:
            # Filtrer les données selon le type sélectionné
            if data_type == 'main':
                dfData = dfData.between_time(
                    share.openMarketTime.strftime('%H:%M'),
                    share.closeMarketTime.strftime('%H:%M')
                )

            # Normalisation si demandée
            if normalize and dfData['openPrice'][0] > 0:
                dfData['openPrice'] /= dfData['openPrice'][0]

            # Données historiques
            fig.add_scatter(
                name=f"{values[i]} (Historical)",
                x=dfData.index,
                y=dfData['openPrice'],
                mode='lines',
                line=dict(color='#4CAF50')
            )

            # Ajouter prédiction si le bouton a été cliqué
            if n_clicks > 0:
                try:
                    testPredict, testX = ls.compute_lstm(share, dfData)
                    if normalize:
                        testPredict = testPredict / dfData['openPrice'][0]
                    
                    fig.add_scatter(
                        name=f"{values[i]} (Prediction)",
                        x=dfData.index[-len(testPredict):],
                        y=testPredict.flatten(),
                        mode='lines',
                        line=dict(
                            color='#FFA500',
                            dash='dash'
                        )
                    )
                except Exception as e:
                    print(f"Prediction error for {values[i]}: {str(e)}")

    # Appliquer le zoom précédent si disponible
    if relayoutData and 'xaxis.range[0]' in relayoutData:
        fig.update_layout(
            xaxis_range=[
                relayoutData['xaxis.range[0]'],
                relayoutData['xaxis.range[1]']
            ]
        )

    return fig
