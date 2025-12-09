# config.py
import sys
import io
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from app import app, shM, socketio
from web.components.navigation import create_navigation, create_page_help

import Models.Shares as sm
import threading
import logging
import os
from Models.SqlCom import SqlCom
from dotenv import load_dotenv
import json

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

# Styles communs
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '24px'
}

INPUT_STYLE = {
    'width': '100%',
    'padding': '12px 16px',
    'backgroundColor': '#12121a',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'borderRadius': '10px',
    'color': '#f8fafc',
    'fontSize': '0.9375rem',
    'fontFamily': 'Outfit, sans-serif',
    'marginBottom': '16px'
}

LABEL_STYLE = {
    'display': 'block',
    'fontSize': '0.8125rem',
    'fontWeight': '500',
    'color': '#94a3b8',
    'marginBottom': '6px',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em'
}

BUTTON_PRIMARY = {
    'background': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
    'color': 'white',
    'padding': '12px 32px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '0.9375rem',
    'fontWeight': '600',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 15px rgba(99, 102, 241, 0.3)',
    'marginTop': '8px'
}

BUTTON_SECONDARY = {
    'background': 'linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)',
    'color': 'white',
    'padding': '12px 32px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '0.9375rem',
    'fontWeight': '600',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 15px rgba(59, 130, 246, 0.3)',
    'marginTop': '8px'
}

help_text = """
### Configuration

Cette page permet de configurer les paramètres de connexion au broker et à la base de données.

#### Paramètres Broker
*   **Type** : Sélectionnez votre broker (Trading 212, etc.).
*   **Credentials** : Entrez vos identifiants de connexion.
*   **Connexion DB** : Paramètres de connexion à la base de données.

#### Export
*   **Export Path** : Chemin où seront sauvegardés les fichiers exportés.
"""

layout = html.Div([
    create_page_help("Aide Configuration", help_text),
    
    # En-tête de page
    html.Div([
        html.H3('Configuration', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Paramètres du système et connexions', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0'
        })
    ], style={'marginBottom': '32px'}),

    # Configuration Broker
    html.Div([
        html.Div([
            html.Span('🔗', style={'fontSize': '1.25rem'}),
            html.Span('Configuration Broker', style={
                'fontSize': '1.125rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '10px'
            })
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Grid de formulaire
        html.Div([
            # Broker Type
            html.Div([
                html.Label('Type de Broker', style=LABEL_STYLE),
                dcc.Dropdown(
                    id='broker_type',
                    options=[
                        {'label': '🏦 Trading 212', 'value': 'trading_212'},
                        {'label': '🏛 Broker A', 'value': 'broker_a'}
                    ],
                    value=config_data.get("broker_type", "broker_a"),
                    style={'marginBottom': '16px'},
                    persistence=True,
                    persistence_type='session'
                )
            ], style={'gridColumn': '1 / -1'}),
            
            # Username
            html.Div([
                html.Label('Nom d\'utilisateur', style=LABEL_STYLE),
                dcc.Input(
                    id='broker_username', 
                    type='text', 
                    value=config_data.get("broker_username", ""),
                    placeholder='Entrez votre username...',
                    style=INPUT_STYLE
                )
            ]),
            
            # Password
            html.Div([
                html.Label('Mot de passe', style=LABEL_STYLE),
                dcc.Input(
                    id='broker_password', 
                    type='password', 
                    value=config_data.get("broker_password", ""),
                    placeholder='••••••••',
                    style=INPUT_STYLE
                )
            ]),
            
            # Host
            html.Div([
                html.Label('Host', style=LABEL_STYLE),
                dcc.Input(
                    id='broker_host', 
                    type='text', 
                    value=config_data.get("broker_host", "localhost"),
                    style=INPUT_STYLE
                )
            ]),
            
            # Port
            html.Div([
                html.Label('Port', style=LABEL_STYLE),
                dcc.Input(
                    id='broker_port', 
                    type='number', 
                    value=config_data.get("broker_port", 5432),
                    style=INPUT_STYLE
                )
            ]),
            
            # Database
            html.Div([
                html.Label('Base de données', style=LABEL_STYLE),
                dcc.Input(
                    id='broker_database', 
                    type='text', 
                    value=config_data.get("broker_database", "stocksprices"),
                    style=INPUT_STYLE
                )
            ]),
            
            # Export Path
            html.Div([
                html.Label('Chemin d\'export', style=LABEL_STYLE),
                dcc.Input(
                    id='export_path', 
                    type='text', 
                    value=config_data.get("export_path", os.getcwd()),
                    style=INPUT_STYLE
                )
            ]),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
            'gap': '16px'
        }),
        
        # Bouton de sauvegarde
        html.Div([
            html.Button(
                '💾 Sauvegarder la configuration', 
                id='save_config', 
                n_clicks=0,
                style=BUTTON_PRIMARY
            ),
            html.Div(id='config_status', style={
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500'
            })
        ], style={'marginTop': '16px'})
    ], style={
        **CARD_STYLE,
        'maxWidth': '800px',
        'margin': '0 auto 24px'
    }),

    # Section Export
    html.Div([
        html.Div([
            html.Span('📦', style={'fontSize': '1.25rem'}),
            html.Span('Export de la Base de Données', style={
                'fontSize': '1.125rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '10px'
            })
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        
        html.Div([
            html.Label('Chemin d\'export', style=LABEL_STYLE),
            dcc.Input(
                id='export_path_input', 
                type='text', 
                value=config_data.get("export_path", os.getcwd()), 
                style={**INPUT_STYLE, 'marginBottom': '0'}
            ),
        ]),
        
        html.Div([
            html.Button(
                '📤 Exporter les données', 
                id='export_button', 
                n_clicks=0,
                style=BUTTON_SECONDARY
            ),
            html.Div(id='export_status', style={
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500'
            })
        ], style={'marginTop': '16px'})
    ], style={
        **CARD_STYLE,
        'maxWidth': '800px',
        'margin': '0 auto 24px'
    }),

    # Spacer pour navigation
    html.Div(style={'height': '100px'}),

    # Navigation
    create_navigation()
], style={
    'backgroundColor': '#0a0a0f',
    'minHeight': '100vh',
    'padding': '24px'
})

@app.callback(
    Output('config_status', 'children'),
    Output('config_status', 'style'),
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
        try:
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
            return "✅ Configuration sauvegardée avec succès !", {
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500',
                'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                'color': '#10b981',
                'border': '1px solid rgba(16, 185, 129, 0.3)'
            }
        except Exception as e:
            return f"❌ Erreur: {str(e)}", {
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500',
                'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                'color': '#ef4444',
                'border': '1px solid rgba(239, 68, 68, 0.3)'
            }
    return "", {}

@app.callback(
    Output('export_status', 'children'),
    Output('export_status', 'style'),
    Input('export_button', 'n_clicks'),
    State('export_path_input', 'value')
)
def export_database(n_clicks, export_path):
    if n_clicks > 0:
        try:
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

            return f"✅ {export_message}", {
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500',
                'backgroundColor': 'rgba(16, 185, 129, 0.1)',
                'color': '#10b981',
                'border': '1px solid rgba(16, 185, 129, 0.3)'
            }
        except Exception as e:
            return f"❌ Erreur: {str(e)}", {
                'marginTop': '16px',
                'padding': '12px',
                'borderRadius': '8px',
                'fontWeight': '500',
                'backgroundColor': 'rgba(239, 68, 68, 0.1)',
                'color': '#ef4444',
                'border': '1px solid rgba(239, 68, 68, 0.3)'
            }
    return "", {}
