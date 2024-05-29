import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from appMain import app
from apps import dashboard, update, prediction #update#, prediction


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' or pathname == '/dashboard':
        return dashboard.layout
    elif pathname == '/update':
        return update.layout
    elif pathname == '/prediction':
        return prediction.layout
    else:
        return '404'
