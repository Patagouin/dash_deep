#index.py
import multiprocessing
import os

# IMPORTANT: Pour utiliser CUDA dans les background workers,
# il faut utiliser la méthode de démarrage 'spawn'.
# Sinon, le 'fork' par défaut hérite d'un contexte CUDA initialisé
# et provoque des erreurs "CUDA_ERROR_NOT_INITIALIZED".
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# IMPORTANT: Ne PAS initialiser TensorFlow dans le processus parent
# car cela cause PyExceptionRegistry::Init() already called dans les workers spawn.
# TensorFlow sera initialisé dans chaque worker spawn avec la bonne config CUDA.
# import web.tf_setup  # Configuration GPU avant tout import TensorFlow
from dash.dependencies import Input, Output  # Ajout des imports manquants
from web.apps import dashboard, update, prediction, config, transaction  # Import the new config page
from web.apps import analyse
from web.apps import simulation
from web.apps import visualisation
from web.apps import playground_callbacks as playground
from dash import dcc, html
from app import app

# Define the layout explicitly
layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Ensure callbacks are defined after the layout
print("Registering callback for page-content")  # Debugging print
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    print("Current pathname:", pathname)  # Debugging print
    if pathname == '/' or pathname == '/dashboard':
        print("Loading dashboard layout")  # Debugging print
        return dashboard.layout
    elif pathname == '/visualisation':
        print("Loading visualisation layout")
        return visualisation.layout
    elif pathname == '/update':
        print("Loading update layout")  # Debugging print
        return update.layout
    elif pathname == '/prediction':
        print("Loading prediction layout")  # Debugging print
        return prediction.layout
    elif pathname == '/analyse':
        print("Loading analyse layout")
        return analyse.layout
    elif pathname == '/simulation':
        print("Loading simulation layout")
        return simulation.layout
    elif pathname == '/playground':
        print("Loading playground layout")
        return playground.layout
    elif pathname == '/config':  # Add the config page route
        print("Loading config layout")  # Debugging print
        return config.layout
    elif pathname == '/transaction':  # Nouvelle route pour Transaction
        print("Loading transaction layout")  # Debugging print
        return transaction.layout
    else:
        print("404 Not Found")  # Debugging print
        return html.Div("404 Not Found")
