# socket_handlers.py
from app import socketio
from flask_socketio import emit
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

@socketio.on('connect')
def handle_connect():
    logging.info("Socket connected!")
    socketio.emit('update_terminal', {'output': 'Socket.IO Connected!\n'}, namespace='/')

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Socket disconnected!")

@socketio.on('message')
def handle_message(message):
    logging.info(f"Received message: {message}")
    socketio.emit('update_terminal', {'output': f'Received message: {message}\n'}, namespace='/')

@socketio.on_error_default
def default_error_handler(e):
    logging.error(f"SocketIO error: {str(e)}")
    socketio.emit('update_terminal', {'output': f'Error: {str(e)}\n'}, namespace='/')
