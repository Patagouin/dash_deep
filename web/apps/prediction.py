#prediction.py
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
import datetime
from web.app import app, shM
from web.apps.sharedFig import fig

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
    ], style={'margin': '20px 0', 'textAlign': 'center'}),

    # Normalize switch
    html.Div([
        daq.BooleanSwitch(
            id='boolean_switch_normalize',
            label="Normalize",
            on=False,
            color="#4CAF50",
            labelPosition="top"
        )
    ], style={'margin': '20px 0', 'textAlign': 'center'}),

    # Navigation standardisée
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
    ], style={
        'width': '100%',
        'textAlign': 'center',
        'backgroundColor': 'black',
        'padding': '20px 0',
        'color': 'white'
    })
], style={'backgroundColor': 'black', 'minHeight': '100vh'})

@app.callback(
    Output('prediction_graph', 'figure'),
    [Input('prediction_dropdown', 'value'),
     Input('date_picker_range', 'start_date'),
     Input('date_picker_range', 'end_date'),
     Input('boolean_switch_normalize', 'on'),
     Input('prediction_graph', 'relayoutData'),
     Input('data_type_selector', 'value')]
)
def update_graph(values, start_date, end_date, toNormalize, relayoutData, data_type):
    if values is None or len(values) == 0:
        print("No values selected.")
        fig = go.Figure()
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
    fig = go.Figure()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(hovermode="x unified")
    fig.update_layout(hoverlabel_bgcolor='#00FF00')
    fig.update_layout(hoverlabel_bordercolor='#00FF00')
    fig.update_layout(legend_bgcolor='#00FF00')
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

            # Filter data based on the selected data type
            if data_type == 'main':
                dfData = dfData.between_time(open_time.strftime('%H:%M'), close_time.strftime('%H:%M'))

            if toNormalize and dfData['openPrice'][0] > 0:
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
    
    return fig
