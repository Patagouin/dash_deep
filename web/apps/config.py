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
import json

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example log to test
logging.warning("Logging is configured and ready to use.")

# Chemin vers le fichier de configuration
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config_data.json")

def load_config():
    """
    Charge les données de configuration depuis le fichier JSON.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        # Si le fichier n'existe pas, créer un fichier par défaut
        default_config = {
            "broker_type": "broker_a",
            "broker_username": "",
            "broker_password": "",
            "broker_host": "localhost",
            "broker_port": 5432,
            "broker_database": "stocksprices"
        }
        save_config(default_config)
        return default_config

    with open(CONFIG_FILE_PATH, "r") as file:
        return json.load(file)

def save_config(config_data):
    """
    Sauvegarde les données de configuration dans le fichier JSON.
    """
    with open(CONFIG_FILE_PATH, "w") as file:
        json.dump(config_data, file, indent=4)

# Charger la configuration actuelle
config_data = load_config()

layout = html.Div([
    html.H3('Configuration Broker'),

    # Broker configuration form
    html.Div([
        html.Label('Broker Type'),
        dcc.Dropdown(
            id='broker_type',
            options=[
                {'label': 'Trading 212', 'value': 'trading_212'},
                {'label': 'Broker A', 'value': 'broker_a'}
            ],
            value=config_data.get("broker_type", "broker_a")  # Charger la valeur depuis le fichier
        ),
        html.Br(),
        html.Label('Username'),
        dcc.Input(id='broker_username', type='text', value=config_data.get("broker_username", "")),
        html.Br(),
        html.Label('Password'),
        dcc.Input(id='broker_password', type='password', value=config_data.get("broker_password", "")),
        html.Br(),
        html.Label('Host'),
        dcc.Input(id='broker_host', type='text', value=config_data.get("broker_host", "localhost")),
        html.Br(),
        html.Label('Port'),
        dcc.Input(id='broker_port', type='number', value=config_data.get("broker_port", 5432)),
        html.Br(),
        html.Label('Database'),
        dcc.Input(id='broker_database', type='text', value=config_data.get("broker_database", "stocksprices")),
        html.Br(),
        html.Label('Export Path'),
        dcc.Input(id='export_path', type='text', value=config_data.get("export_path", os.getcwd())),
        html.Br(),
        html.Button('Save Configuration', id='save_config', n_clicks=0),
        html.Div(id='config_status', style={'color': 'green'})
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Hr(),

    # Section pour l'exportation des données
    html.H3('Export Database'),
    html.Div([
        html.Label('Export Path'),
        dcc.Input(id='export_path_input', type='text', value=config_data.get("export_path", os.getcwd()), style={'width': '100%'}),
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
    State('broker_database', 'value'),
    State('export_path', 'value')
)
def save_configuration(n_clicks, broker_type, username, password, host, port, database, export_path):
    if n_clicks > 0:
        # Charger la configuration actuelle
        config_data = load_config()

        # Mettre à jour les valeurs
        config_data.update({
            "broker_type": broker_type,
            "broker_username": username,
            "broker_password": password,
            "broker_host": host,
            "broker_port": port,
            "broker_database": database,
            "export_path": export_path
        })

        # Sauvegarder dans le fichier JSON
        save_config(config_data)

        logging.info(f"Configuration saved: {config_data}")
        return "Configuration saved successfully!"
    return ""

# Callback pour gérer l'exportation des données
@app.callback(
    Output('export_status', 'children'),
    Input('export_button', 'n_clicks'),
    State('export_path_input', 'value')
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
