# app.py
import os
import diskcache
from flask_socketio import SocketIO
import logging
from dash import CeleryManager, DiskcacheManager, Dash
from dash_extensions.enrich import DashProxy, MultiplexerTransform

from Models.Shares import Shares

# --- Background Callback Manager ---
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# --- App Instantiation ---
shM = Shares()
# Use DashProxy for compatibility with MultiplexerTransform and other extensions
app = DashProxy(
    __name__,
    transforms=[MultiplexerTransform()],
    background_callback_manager=background_callback_manager,
    suppress_callback_exceptions=True,
    external_stylesheets=['/assets/style.css']
)
server = app.server
socketio = SocketIO(server, async_mode='threading')

# Réduire les logs verbeux de werkzeug
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app.title = "Dash Deep"
application = app.server

# Import socket handlers at the end to avoid circular imports
from sockets import socket_handlers

