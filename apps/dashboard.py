import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dash_daq as daq

#useless
import dash_bootstrap_components as dbc

import plotly.graph_objs as go

import datetime

from appMain import app, shM



fig = go.FigureWidget()


layout = html.Div([
    html.H3('Dashboard'),
    dcc.Dropdown(
        id='dashboard_dropdown',
        options=[
            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} for stock in shM.listShares.itertuples()
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
    dcc.Link('Go update', href='/update')
])

@app.callback(
    Output('stock_graph', 'figure'),
    Input('dashboard_dropdown', 'value'),
    Input('date_picker_range', 'start_date'),
    Input('date_picker_range', 'end_date'),
    Input('boolean_switch_normalize', 'on')
    )
def display_value(values, start_date, end_date, toNormalize):

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
    listShares = shM.getRowsByKeysValues(['symbol']*len(values),values)

    dfData = shM.getListDfDataFromDf(listShares, dateBegin, dateLast)
    #dfData[0]; [0] because getListDfDataFromDf give an array of df
    if toNormalize and len(dfData) > 0 and dfData[0]['openPrice'][0] > 0: #Avoid divided by zero
        dfData[0]['openPrice'] /= dfData[0]['openPrice'][0]
    fig.add_scatter(name=value, x=dfData[0].index.values, y=dfData[0]['openPrice'].values)

  
    return fig