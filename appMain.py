from flask import Flask
import dash
import flask_socketio
from flask_socketio import SocketIO
#import dash_auth

import users
import os

import dash_html_components as html

from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine

import dash_bootstrap_components as dbc


########### MODEL INSTANCIATION FOR ALL APPS #############
import Models.Shares as sm

shM = sm.Shares(readOnlyThosetoUpdate=False)
########################################################


#cssFiles = "assets/style.css"
server = Flask(__name__)

app = dash.Dash(__name__, server=server)
socketio = SocketIO(server)

@socketio.on('update_textarea')
def handle_message(message):
    socketio.emit('update_textarea', message)