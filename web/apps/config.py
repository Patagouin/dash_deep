# config.py
import sys
import io
import dash  # Ajout de l'import manquant
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, shM, socketio
import Models.Shares as sm
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example log to test
logging.warning("Logging is configured and ready to use.")

layout = html.Div([
    html.H3('Configuration Broker'),

    # Broker configuration form
    html.Div([
        html.Label('Broker Type'),
        dcc.Dropdown(
            id='broker_type',
            options=[
                {'label': 'Broker A', 'value': 'broker_a'},
                {'label': 'Broker B', 'value': 'broker_b'}
            ],
            value='broker_a'  # Default value
        ),
        html.Br(),
        html.Label('Username'),
        dcc.Input(id='broker_username', type='text', value=''),
        html.Br(),
        html.Label('Password'),
        dcc.Input(id='broker_password', type='password', value=''),
        html.Br(),
        html.Label('Host'),
        dcc.Input(id='broker_host', type='text', value='localhost'),
        html.Br(),
        html.Label('Port'),
        dcc.Input(id='broker_port', type='number', value=5432),
        html.Br(),
        html.Label('Database'),
        dcc.Input(id='broker_database', type='text', value='stocksprices'),
        html.Br(),
        html.Button('Save Configuration', id='save_config', n_clicks=0),
        html.Div(id='config_status', style={'color': 'green'})
    ], style={'width': '50%', 'margin': 'auto'}),

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
            dcc.Link('Prediction', href='/prediction', style={'color': '#4CAF50', 'textDecoration': 'none'}),
            html.Span(' | ', style={'margin': '0 10px', 'color': '#666'}),
            dcc.Link('Update', href='/update', style={'color': '#4CAF50', 'textDecoration': 'none'})
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
    Output('config_status', 'children'),
    Input('save_config', 'n_clicks'),
    State('broker_type', 'value'),
    State('broker_username', 'value'),
    State('broker_password', 'value'),
    State('broker_host', 'value'),
    State('broker_port', 'value'),
    State('broker_database', 'value')
)
def save_configuration(n_clicks, broker_type, username, password, host, port, database):
    if n_clicks > 0:
        logging.info(f"Configuration saved: {broker_type}, {username}, {password}, {host}, {port}, {database}")
        return "Configuration saved successfully!"
    return ""
