# socket_handlers.py
from app import socketio
from flask_socketio import emit
import logging
import sys

@socketio.on('connect')
def handle_connect():
    print("Socket connected!", file=sys.stderr)  # Print to stderr for immediate output
    socketio.emit('update_terminal', {'output': 'Socket.IO Connected!\n'}, namespace='/')

@socketio.on('disconnect')
def handle_disconnect():
    print("Socket disconnected!", file=sys.stderr)

@socketio.on('message')
def handle_message(message):
    print(f"Received message: {message}", file=sys.stderr)
    socketio.emit('update_terminal', {'output': f'Received message: {message}\n'}, namespace='/')

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO error: {str(e)}", file=sys.stderr)
    socketio.emit('update_terminal', {'output': f'Error: {str(e)}\n'}, namespace='/')
