# app.py
import sys
import os

# Ajout du répertoire racine au sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
import dash
from flask_socketio import SocketIO

# Importation du module Shares (ou autre module contenant shM)
from Models import Shares as sm  # Remplacez par le chemin correct vers votre module

# Initialisation du serveur Flask
server = Flask(__name__)

# Initialisation de l'application Dash
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True  # Add this line
)
# Initialisation de SocketIO
socketio = SocketIO(server)

# Initialisation de l'objet shM (par exemple, une instance de la classe Shares)
shM = sm.Shares(readOnlyThosetoUpdate=False)  # Remplacez par l'initialisation correcte

@socketio.on('update_textarea')
def handle_message(message):
    socketio.emit('update_textarea', message)