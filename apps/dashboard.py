import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objs as go

import datetime

from appMain import app, shM
from .sharedFig import fig


layout = html.Div([
    html.H3('Dashboard'),
    dcc.Dropdown(
        id='dashboard_dropdown',
        options=[
            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} for stock in shM.dfShares.itertuples()
        ],
        multi=True
    ),
    html.Div([
        html.Div([
            dcc.Graph(id='stock_graph', figure=fig)
        ], style={'width': '90%', 'float' : 'left'}),
        html.Div([
            daq.BooleanSwitch(id='boolean_switch_normalize', label="Normalize", on=False, color="#f80")
        ], style={'width': '10%', 'float' : 'right'}),
    ], style={'width': '100%'}),
    dcc.DatePickerRange(
        id='date_picker_range',
        display_format='DD/MM/YY',
        start_date=datetime.datetime.now()-datetime.timedelta(days=7),
        end_date=datetime.datetime.now()
    ),
    html.Br(),
    dcc.Link('Go update', href='/update'),
    html.Br(),
    dcc.Link('Go to prediction', href='/prediction')
])

@app.callback(
    Output('stock_graph', 'figure'),
    Input('dashboard_dropdown', 'value'),
    Input('date_picker_range', 'start_date'),
    Input('date_picker_range', 'end_date'),
    Input('boolean_switch_normalize', 'on')
    )
def display_stock_graph(values, start_date, end_date, toNormalize):

    if values is None:
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    
    if end_date != None:
        dateLast = datetime.datetime.fromisoformat(end_date)
    else:
        dateLast = datetime.datetime.now()

    if start_date != None:
        dateBegin = datetime.datetime.fromisoformat(start_date)
    else:
        dateBegin = dateLast-datetime.timedelta(days=7)


    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(hovermode="x unified")
    fig.update_layout(hoverlabel_bgcolor='#00FF00')
    fig.update_layout(hoverlabel_bordercolor='#00FF00')
    fig.update_layout(legend_bgcolor='#00FF00')
    #fig.update_layout(xaxis_tickformat = '%d %B<br>%Y')
    fig.layout.title = 'Shares quots'
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values),values, op = '|')

    listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)
    #dfData[0]; [0] because getListDfDataFromDf give an array of df
    for i, dfData in enumerate(listDfData):
        if not dfData.empty:
            if toNormalize and dfData['openPrice'][0] > 0:#Avoid divided by zero
                dfData['openPrice'] /= dfData['openPrice'][0]
            #fig.add_scatter(name="Quotations", x=dfData[0].index.values, y=dfData[0]['openPrice'].values)
            fig.add_scatter(name=values[i], x=dfData.index.values, y=dfData['openPrice'].values)

  
    return fig