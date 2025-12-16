# app.py
import os
# IMPORTANT: Ne PAS initialiser TensorFlow dans le processus parent
# car cela cause PyExceptionRegistry::Init() already called dans les workers spawn.
# TensorFlow sera initialisé dans chaque worker spawn avec la bonne config CUDA.
# try:
#     import web.tf_setup  # noqa: F401
# except Exception:
#     pass
import diskcache
from flask_socketio import SocketIO
import logging
from dash_extensions.enrich import DashProxy, MultiplexerTransform

from Models.Shares import Shares
from web.custom_diskcache_manager import SpawnDiskcacheManager

# --- Background Callback Manager ---
# Utiliser SpawnDiskcacheManager qui utilise 'spawn' au lieu de 'fork'
# Cela permet d'utiliser CUDA dans les background callbacks sans avoir besoin de Celery/Redis
cache = diskcache.Cache("./cache")
background_callback_manager = SpawnDiskcacheManager(cache)
logging.info("✅ DiskcacheManager avec 'spawn' activé pour support GPU dans les background callbacks")
logging.info("   (Pas besoin de Celery/Redis)")

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

