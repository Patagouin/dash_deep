# config.py
import sys
import io
import dash  # Ajout de l'import manquant
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, shM, socketio
from web.components.navigation import create_navigation  # Importer le composant de navigation

import Models.Shares as sm
import threading
import logging
import os  # Importer os pour gérer les chemins de fichiers
from Models.SqlCom import SqlCom  # Importer la classe SqlCom
from dotenv import load_dotenv  # Add this to load environment variables

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
                {'label': 'Trading 212', 'value': 'trading_212'}  # Ajout de Trading 212
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

    html.Hr(),

    # Section pour l'exportation des données
    html.H3('Export Database'),
    html.Div([
        html.Label('Export Path'),
        dcc.Input(id='export_path', type='text', value=os.getcwd(), style={'width': '100%'}),  # Chemin par défaut = répertoire courant
        html.Br(),
        html.Button('Export Data', id='export_button', n_clicks=0),
        html.Div(id='export_status', style={'color': 'green'})
    ], style={'width': '50%', 'margin': 'auto'}),

    # Navigation standardisée avec séparateur
    create_navigation()
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
        if broker_type == 'trading_212':
            # Sauvegarder les informations spécifiques à Trading 212
            logging.info(f"Trading 212 configuration saved: {username}, {password}")
            # Vous pouvez ajouter ici la logique pour sauvegarder ces informations dans un fichier ou une base de données
            return "Trading 212 configuration saved successfully!"
        else:
            logging.info(f"Configuration saved: {broker_type}, {username}, {password}, {host}, {port}, {database}")
            return "Configuration saved successfully!"
    return ""

# Callback pour gérer l'exportation des données
@app.callback(
    Output('export_status', 'children'),
    Input('export_button', 'n_clicks'),
    State('export_path', 'value')
)
def export_database(n_clicks, export_path):
    if n_clicks > 0:
        # Créer une instance de SqlCom pour interagir avec la base de données
        sql_com = SqlCom(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            sharesObj=None
        )

        # Appeler la méthode d'exportation
        export_message = sql_com.export_data_to_csv(export_path)

        return export_message
    return ""
