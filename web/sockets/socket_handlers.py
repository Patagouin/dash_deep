# socket_handlers.py
from app import socketio
from flask_socketio import emit
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

@socketio.on('connect')
def handle_connect(sid, environ):
    logging.info(f"Socket connected: {sid}")
    socketio.emit('update_terminal', {'output': 'Socket.IO Connected!\n'}, namespace='/', room=sid)

@socketio.on('disconnect')
def handle_disconnect(sid):
    logging.info(f"Socket disconnected: {sid}")

@socketio.on('message')
def handle_message(sid, message):
    logging.info(f"Received message from {sid}: {message}")
    socketio.emit('update_terminal', {'output': f'Received message: {message}\n'}, namespace='/', room=sid)

@socketio.on_error_default
def default_error_handler(e):
    logging.error(f"SocketIO error: {e}")
    # It's good practice to also handle the disconnection or notify the client
    # For now, we just log it. You might want to add more logic here.
    pass
