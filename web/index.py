#index.py
from dash.dependencies import Input, Output  # Ajout des imports manquants
from web.apps import dashboard, update, prediction, config, transaction  # Import the new config page
from web.apps import analyse
from web.apps import simulation
from web.apps import visualisation
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
    elif pathname == '/config':  # Add the config page route
        print("Loading config layout")  # Debugging print
        return config.layout
    elif pathname == '/transaction':  # Nouvelle route pour Transaction
        print("Loading transaction layout")  # Debugging print
        return transaction.layout
    else:
        print("404 Not Found")  # Debugging print
        return html.Div("404 Not Found")
