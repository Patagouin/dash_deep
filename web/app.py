# app.py
import sys
import os
from flask import Flask, send_from_directory
import dash
from flask_socketio import SocketIO
from Models import Shares as sm

# Add the root directory to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Initialize the Flask server
server = Flask(__name__)

# Initialize SocketIO with CORS settings and debug mode
socketio = SocketIO(
    server,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,  # Désactiver le logger SocketIO
    engineio_logger=False,  # Désactiver le logger Engine.IO
    ping_timeout=60,
    ping_interval=25,
    always_connect=True,
    debug=False  # Désactiver le mode debug
)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
)

# Serve static files
@server.route('/assets/<path:path>')
def serve_static(path):
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    return send_from_directory(assets_dir, path)

# Initialize the shM object
shM = sm.Shares(readOnlyThosetoUpdate=False)

# Expose the server variable for WSGI servers
application = app.server

# Import socket handlers
from sockets import socket_handlers

if __name__ == '__main__':
    socketio.run(app, debug=True)
