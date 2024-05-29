import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objs as go

import datetime

from appMain import app, shM
from .sharedFig import fig
import deepLearning.lstm as ls



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
    html.Br(),
    dcc.Link('Go update', href='/update'),
    html.Br(),
    dcc.Link('Go to dashboard', href='/dashboard')
])

@app.callback(
    Output('prediction_stock_graph', 'figure'),
    Input('dashboard_dropdown', 'value'),
    Input('boolean_switch_normalize', 'on')
    )
def display_prediction_stock_graph(values, toNormalize):

    if values is None:
        fig = go.FigureWidget()
        fig.layout.template = 'plotly_dark'
        return fig
    

    fig = go.FigureWidget()
    fig.layout.template = 'plotly_dark'
    fig.update_layout(hovermode="x unified")
    fig.update_layout(hoverlabel_bgcolor='#00FF00')
    fig.update_layout(hoverlabel_bordercolor='#00FF00')
    fig.update_layout(legend_bgcolor='#00FF00')
    #fig.update_layout(xaxis_tickformat = '%d %B<br>%Y')
    fig.layout.title = 'Shares quots'
    dfShares = shM.getRowsDfByKeysValues(['symbol']*len(values),values, op = '|')
    #
    for curShare in dfShares.itertuples():
        data_quots = self.get_cotations_data_df(curShare, curShare.firstRecord, curShare.lastRecord)
        testPredict, testX = ls.compute_lstm(curShare, data_quots)
    #
    #listDfData = shM.getListDfDataFromDfShares(dfShares, dateBegin, dateLast)
    #dfData[0]; [0] because getListDfDataFromDf give an array of df
    for i, dfData in enumerate(listDfData):
        if not dfData.empty:
            if toNormalize and dfData['openPrice'][0] > 0:#Avoid divided by zero
                dfData['openPrice'] /= dfData['openPrice'][0]
            #fig.add_scatter(name="Quotations", x=dfData[0].index.values, y=dfData[0]['openPrice'].values)
            fig.add_scatter(name=values[i], x=dfData.index.values, y=dfData['openPrice'].values)

  
    return fig