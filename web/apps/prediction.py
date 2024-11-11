#prediction.py
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
from web.components.navigation import create_navigation  # Importer le composant de navigation

import datetime

from app import app, shM

# Remplacer l'importation de fig par la création directe de la figure
fig = go.FigureWidget()

from Models.trading212 import buy_stock, sell_stock

layout = html.Div([
    # Conteneur fixe pour le bandeau supérieur
    html.Div([
        html.H3('Prediction'),
        
        # Ligne de contrôles supérieurs
        html.Div([
            # Date picker
            html.Div([
                dcc.DatePickerRange(
                    id='date_picker_range',
                    display_format='DD/MM/YY',
                    start_date=datetime.datetime.now()-datetime.timedelta(days=7),
                    end_date=datetime.datetime.now()
                ),
            ], style={'display': 'inline-block'}),
            
            # Radio items pour All Data/Main Hours
            html.Div([
                dcc.RadioItems(
                    id='data_type_selector',
                    options=[
                        {'label': 'All Data', 'value': 'all'},
                        {'label': 'Main Hours', 'value': 'main'}
                    ],
                    value='all',
                    inline=True,
                    labelStyle={'marginRight': '20px'},
                    style={'color': '#4CAF50'}
                )
            ], style={'display': 'inline-block', 'marginLeft': '20px'}),

            # Checkbox pour Normalize
            html.Div([
                dcc.Checklist(
                    id='normalize_checkbox',
                    options=[{'label': 'Normalize', 'value': 'normalize'}],
                    value=[],
                    inline=True,
                    labelStyle={'color': '#4CAF50', 'marginRight': '20px'}
                ),
            ], style={'display': 'inline-block', 'marginLeft': '20px'}),

            # Dropdown pour la sélection des actions
            html.Div([
                dcc.Dropdown(
                    id='prediction_dropdown',
                    options=[
                        {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
                        for stock in sorted(shM.dfShares.itertuples(), key=lambda stock: stock.symbol)
                    ],
                    multi=True,
                    placeholder="Sélectionner une action",
                    style={'width': '200px'}
                ),
            ], style={'display': 'inline-block', 'marginLeft': '20px'})
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '20px'
        })
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'backgroundColor': 'black',
        'padding': '20px',
        'zIndex': 1000
    }),

    # Conteneur défilable pour le reste du contenu
    html.Div([
        # Graph container
        html.Div([
            dcc.Graph(
                id='stock_graph',
                figure=fig,
                config={'scrollZoom': True},
                style={
                    'backgroundColor': 'black',
                    'height': '50vh'
                }
            )
        ], style={
            'width': '100%',
            'margin': '20px 0',
            'backgroundColor': 'black',
            'padding': '20px 0'
        }),

        # Section des paramètres du modèle
        html.Div([
            html.H4('Paramètres du Modèle'),

            # Container pour tous les contrôles
            html.Div([
                # Première ligne : paramètres de base
                html.Div([
                    # Colonne Look Back X
                    html.Div([
                        html.Label('Look Back X'),
                        dcc.Input(
                            id='look_back_x',
                            type='number',
                            value=60,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Stride X
                    html.Div([
                        html.Label('Stride X'),
                        dcc.Input(
                            id='stride_x',
                            type='number',
                            value=2,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Nombre de Prédictions
                    html.Div([
                        html.Label('Nombre de Prédictions'),
                        dcc.Input(
                            id='nb_y',
                            type='number',
                            value=1,
                            min=1,
                            style={'width': '100%'}
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),

                # Deuxième ligne : hyperparamètres
                html.Div([
                    # Colonne Nombre d'Unités
                    html.Div([
                        html.Label('Nombre d\'Unités (multiple)'),
                        dcc.Dropdown(
                            id='nb_units',
                            options=[{'label': str(units), 'value': units} for units in [32, 64, 128, 256, 512]],
                            value=[64, 128],
                            multi=True
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Nombre de Couches
                    html.Div([
                        html.Label('Nombre de Couches (multiple)'),
                        dcc.Dropdown(
                            id='layers',
                            options=[{'label': str(layers), 'value': layers} for layers in [1, 2, 3]],
                            value=[1, 2],
                            multi=True
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),

                # Troisième ligne : paramètres d'apprentissage
                html.Div([
                    # Colonne Taux d'Apprentissage
                    html.Div([
                        html.Label('Taux d\'Apprentissage (multiple)'),
                        dcc.Dropdown(
                            id='learning_rate',
                            options=[{'label': str(lr), 'value': lr} for lr in [1e-2, 5e-3, 1e-3]],
                            value=[1e-3, 5e-3],
                            multi=True
                        ),
                    ], style={'flex': '1', 'marginRight': '20px'}),

                    # Colonne Fonction de Perte
                    html.Div([
                        html.Label('Fonction de Perte'),
                        dcc.Dropdown(
                            id='loss_function',
                            options=[
                                {'label': 'Mean Squared Error', 'value': 'mean_squared_error'},
                                {'label': 'Mean Absolute Error', 'value': 'mean_absolute_error'},
                            ],
                            value='mean_squared_error'
                        ),
                    ], style={'flex': '1'}),
                ], style={
                    'display': 'flex',
                    'marginBottom': '20px',
                    'gap': '10px'
                }),
            ], style={
                'width': '100%',
                'padding': '20px'
            }),

            # Bouton d'entraînement
            html.Div([
                html.Button(
                    'Entraîner le Modèle',
                    id='train_button',
                    n_clicks=0,
                    style={
                        'padding': '10px 20px',
                        'fontSize': '16px',
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer'
                    }
                )
            ], style={
                'textAlign': 'center',
                'width': '100%'
            }),
        ], style={
            'width': '100%',
            'padding': '20px',
            'backgroundColor': '#1E1E1E',
            'borderRadius': '8px',
            'marginBottom': '20px'
        }),

        create_navigation()
    ], style={
        'marginTop': '120px',  # Espace pour le bandeau fixe
        'padding': '20px',
        'backgroundColor': 'black',
        'minHeight': 'calc(100vh - 120px)',  # Hauteur moins l'espace du bandeau
        'overflowY': 'auto'  # Permet le défilement
    })
], style={
    'backgroundColor': 'black',
    'minHeight': '100vh'
})

@app.callback(
    Output('stock_graph', 'figure'),  # ID of the output component
    Input('prediction_dropdown', 'value'),  # ID of the dropdown
    Input('date_picker_range', 'start_date'),  # ID of the date picker (start date)
    Input('date_picker_range', 'end_date'),  # ID of the date picker (end date)
    Input('data_type_selector', 'value'),  # Capture the selected data type (All Data or Main Hours)
    Input('normalize_checkbox', 'value'),  # Nouveau input pour la checkbox
    Input('stock_graph', 'relayoutData'),  # Capture the current layout (zoom level)
    Input('train_button', 'n_clicks'),  # Bouton pour entraîner le modèle
    State('look_back_x', 'value'),
    State('stride_x', 'value'),
    State('nb_y', 'value'),
    State('nb_units', 'value'),
    State('layers', 'value'),
    State('learning_rate', 'value'),
    State('loss_function', 'value'),
)
def display_stock_graph(values, start_date, end_date, data_type, normalize_value, relayoutData,
                       n_clicks, look_back_x, stride_x, nb_y, nb_units, layers, learning_rate, loss_function):

    if values is None or len(values) == 0:
        print("No values selected.")
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    
    # Conversion des dates de début et de fin
    if end_date is not None:
        dateLast = datetime.datetime.fromisoformat(end_date)
    else:
        dateLast = datetime.datetime.now()

    if start_date is not None:
        dateBegin = datetime.datetime.fromisoformat(start_date)
    else:
        dateBegin = dateLast - datetime.timedelta(days=7)

    print("Date range:", dateBegin, "to", dateLast)

    # Création d'une figure vide avec un thème sombre
    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(hovermode="x unified")
    fig.update_layout(hoverlabel=dict(
        bgcolor='#FFFFFF',  # Fond blanc
        font=dict(color='#000000'),  # Texte noir
        bordercolor='#FFFFFF'  # Bordure blanche
    ))
    fig.update_layout(
        legend=dict(
            font=dict(color='#FFFFFF'),  # Texte de légende en blanc
            bgcolor='rgba(0,0,0,0.5)',   # Fond semi-transparent
            bordercolor='#FFFFFF'         # Bordure blanche
        )
    )
    fig.layout.title = 'Shares quots'
    
    # Récupération des données des actions sélectionnées
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values), values, op='|')
    print("DataFrame for selected values:", dfShares)

    if dfShares.empty:
        print("No data found for the selected symbols.")
        return fig

    # Récupération des données pour chaque action dans la plage de dates sélectionnée
    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)
    print("List of DataFrames:", listDfData)

    # Ajout des données au graphique
    for i, dfData in enumerate(listDfData):
        if not dfData.empty:
            print(f"Data for {values[i]}:", dfData)

            # Get the open and close market times for the current stock
            open_time = dfShares.iloc[i].openMarketTime
            close_time = dfShares.iloc[i].closeMarketTime

            # Filter data based on the selected data type (All Data or Main Hours)
            if data_type == 'main':
                # Filter out extended hours data using the stock's specific market hours
                dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))

            # Mise à jour de la logique de normalisation
            toNormalize = 'normalize' in normalize_value

            if toNormalize and dfData['openPrice'][0] > 0:  # Normalisation si nécessaire
                dfData['openPrice'] /= dfData['openPrice'][0]
            fig.add_scatter(name=values[i], x=dfData.index.values, y=dfData['openPrice'].values)
        else:
            print(f"No data for {values[i]}")

    # Apply the previous zoom level if available
    if relayoutData and 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
        fig.update_layout(
            xaxis_range=[relayoutData['xaxis.range[0]'], relayoutData['xaxis.range[1]']]
        )

        # Filter data based on the zoomed x-axis range
        x_min = relayoutData['xaxis.range[0]']
        x_max = relayoutData['xaxis.range[1]']

        # Find the min and max y-values within the zoomed x-axis range
        y_min, y_max = None, None
        for dfData in listDfData:
            zoomed_data = dfData[(dfData.index >= x_min) & (dfData.index <= x_max)]
            if not zoomed_data.empty:
                current_y_min = zoomed_data['openPrice'].min()
                current_y_max = zoomed_data['openPrice'].max()
                if y_min is None or current_y_min < y_min:
                    y_min = current_y_min
                if y_max is None or current_y_max > y_max:
                    y_max = current_y_max

        # Update the y-axis range based on the zoomed data
        if y_min is not None and y_max is not None:
            fig.update_layout(
                yaxis_range=[y_min, y_max]
            )

    # # Préparation des dictionnaires data_info et hps
    # data_info = {
    #     'symbol': values[0] if values else None,
    #     'look_back_x': look_back_x,
    #     'stride_x': stride_x,
    #     'nb_y': nb_y,
    #     'percent_train_test': 60,
    #     'nb_days_to_take_dataset': 400,
    #     'features': ['openPrice'],
    # }

    # hps = {
    #     'nb_units': nb_units,  # Maintenant une liste de valeurs
    #     'layers': layers,      # Maintenant une liste de valeurs
    #     'epochs': 50,
    #     'epochs_tuner': 10,
    #     'batch_size': 16,
    #     'learning_rate': learning_rate,  # Maintenant une liste de valeurs
    #     'loss': loss_function,
    # }

    # # Chargement des données et préparation
    # from Models import prediction_utils as pu
    # shareObj = shM.dfShares[shM.dfShares['symbol'] == data_info['symbol']].iloc[0]
    # data_info['shareObj'] = shareObj

    # df = pu.get_and_clean_data(shM, shareObj, columns=data_info['features'])
    # trainX, trainY, testX, testY = pu.create_train_test(df, data_info)

    # # Entraînement du modèle uniquement si le bouton est cliqué
    # if n_clicks > 0:
    #     # Entraîner le modèle et obtenir les prédictions
    #     model, history = pu.train_and_select_best_model(data_info, hps, trainX, trainY, testX, testY)

    #     # Préparation des données pour la prédiction
    #     X_pred = testX[-1:]  # Dernier échantillon de test
    #     y_pred = model.predict(X_pred)

    #     # Convertir y_pred en valeurs originales si nécessaire
    #     # Ajouter les prédictions au graphique
    #     future_times = [df.index[-1] + datetime.timedelta(minutes=data_info['stride_x'] * (i+1)) for i in range(nb_y)]
    #     fig.add_scatter(
    #         x=future_times,
    #         y=y_pred.flatten(),
    #         mode='markers',
    #         name='Prédictions Futures',
    #         marker=dict(color='red', size=10)
    #     )

    return fig
print("App object in prediction.py:", id(app))
