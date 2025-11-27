# main.py
import sys
import os
from dotenv import load_dotenv  # Importation de python-dotenv
from flask_socketio import SocketIO  # Add this import
from flask import Flask

# Ajout du répertoire racine au sys.path de manière plus robuste
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Charger les variables d'environnement à partir d'un chemin spécifique
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from app import app, socketio  # Import socketio from app.py
from index import layout  # Importation du layout depuis `index.py`

# Assigner le layout à l'application Dash
app.layout = layout

if __name__ == '__main__':
    socketio.run(
        app.server,
        debug=False,  # Désactiver le mode debug
        port=8050,
        allow_unsafe_werkzeug=True,
        log_output=False  # Désactiver les logs de sortie
    )
