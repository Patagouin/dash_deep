from dash import dcc, html
from app import shM
from web.components.navigation import create_navigation
from web.apps.prediction_visualization import get_visualization_layout
# Importer les callbacks du graphe pour enregistrer les callbacks nécessaires
import web.apps.prediction_callbacks.graph  # noqa: F401


def _shares_dropdown():
    return html.Div([
        html.Label('Choix actions'),
        dcc.Dropdown(
            id='train_share_list',
            options=[
                *[
                    {'label': f"{stock.symbol}", 'value': stock.symbol}
                    for stock in shM.dfShares.sort_values(by='symbol').itertuples()
                ]
            ],
            multi=True,
            placeholder="Choisir une ou plusieurs actions",
            style={'width': '100%', 'color': '#FF8C00'},
            persistence=True, persistence_type='session'
        )
    ], style={'backgroundColor': '#2E2E2E', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '10px'})


layout = html.Div([
    html.H3('Visualisation', style={'color': '#FF8C00'}),
    _shares_dropdown(),
    get_visualization_layout(),
    create_navigation()
], style={'backgroundColor': 'black', 'minHeight': '100vh', 'padding': '20px'})


