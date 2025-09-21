from dash import dcc, html
from web.components.navigation import create_navigation

# Import the new sections
from web.apps.prediction_parameters import get_parameters_layout
from web.apps.prediction_results import get_results_layout
from web.apps.prediction_visualization import get_visualization_layout

# Import the callbacks package to register all modularized callbacks
import web.apps.prediction_callbacks  # noqa: F401

layout = html.Div([
    # Fixed container for the top banner
    html.Div([
        html.H3('Prediction'),
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'backgroundColor': 'black',
        'padding': '20px',
        'zIndex': 1000
    }),

    # Scrollable container for the rest of the content
    html.Div([
        # Import the "Paramètres du Modèle" section
        get_parameters_layout(),

        # Import the "Résultats du Modèle" section
        get_results_layout(),

        # Import the "Visualisation des Prédictions" section
        get_visualization_layout(),

        create_navigation()
    ], style={
        'marginTop': '80px',
        'padding': '20px',
        'backgroundColor': 'black',
        'minHeight': 'calc(100vh - 80px)',
        'overflowY': 'auto'
    })
], style={
    'backgroundColor': 'black',
    'minHeight': '100vh'
})
