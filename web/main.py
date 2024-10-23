# main.py
import sys
import os
from dotenv import load_dotenv  # Importation de python-dotenv
from flask_socketio import SocketIO  # Add this import

# Charger les variables d'environnement à partir d'un chemin spécifique
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Ajout du répertoire parent au sys.path pour permettre les importations relatives
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app  # Importation de l'objet `app` depuis `app.py`
from index import layout  # Importation du layout depuis `index.py`

# Assigner le layout à l'application Dash
app.layout = layout

# Initialize SocketIO with the Flask server (not the Dash app)
socketio = SocketIO(app.server)

if __name__ == '__main__':
    socketio.run(app.server, debug=True, use_reloader=False)  # Use socketio.run instead of app.run_server
