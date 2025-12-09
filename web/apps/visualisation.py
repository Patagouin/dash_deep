from dash import dcc, html
from app import shM
from web.components.navigation import create_navigation, create_page_help
from web.apps.prediction_visualization import get_visualization_layout
# Importer les callbacks du graphe pour enregistrer les callbacks nécessaires
import web.apps.prediction_callbacks.graph  # noqa: F401

# Styles communs
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '20px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '16px'
}

SECTION_TITLE_STYLE = {
    'fontSize': '0.875rem',
    'fontWeight': '600',
    'color': '#a78bfa',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em',
    'marginBottom': '12px',
    'display': 'flex',
    'alignItems': 'center',
    'gap': '8px'
}

help_text = """
### Visualisation des Données

Cette page vous permet d'explorer visuellement les données historiques de vos actions.

#### Fonctionnalités
*   **Sélection multiple** : Choisissez une ou plusieurs actions pour les comparer.
*   **Graphiques interactifs** : Zoomez, survolez pour voir les détails.
*   **Indicateurs techniques** : Visualisez les moyennes mobiles, volumes, etc.

#### Conseils
*   Comparez des actions du même secteur pour identifier des opportunités.
*   Utilisez le zoom pour analyser des périodes spécifiques.
"""

def _shares_dropdown():
    return html.Div([
        html.Div([
            html.Span('📊', style={'fontSize': '1rem'}),
            html.Span('Sélection des Actions', style=SECTION_TITLE_STYLE)
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'marginBottom': '12px'}),
        dcc.Dropdown(
            id='train_share_list',
            options=[
                *[
                    {'label': f"{stock.symbol}", 'value': stock.symbol}
                    for stock in shM.dfShares.sort_values(by='symbol').itertuples()
                ]
            ],
            multi=True,
            placeholder="Choisir une ou plusieurs actions...",
            style={'width': '100%'},
            persistence=True, 
            persistence_type='session'
        ),
        html.P(
            "Sélectionnez les actions que vous souhaitez visualiser dans les graphiques ci-dessous.",
            style={
                'fontSize': '0.8125rem',
                'color': '#64748b',
                'marginTop': '8px',
                'marginBottom': '0'
            }
        )
    ], style=CARD_STYLE)


layout = html.Div([
    create_page_help("Aide Visualisation", help_text),
    
    # En-tête de page
    html.Div([
        html.H3('Visualisation', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Explorez les données historiques de vos actifs', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0'
        })
    ], style={'marginBottom': '24px'}),
    
    # Dropdown des actions
    _shares_dropdown(),
    
    # Graphiques de visualisation
    html.Div([
        get_visualization_layout()
    ], style={
        **CARD_STYLE,
        'marginBottom': '100px'
    }),
    
    # Navigation
    create_navigation()
], style={
    'backgroundColor': '#0a0a0f',
    'minHeight': '100vh',
    'padding': '24px 32px',
    'width': '100%',
    'maxWidth': '100%',
    'margin': '0'
})
