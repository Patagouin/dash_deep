from dash import dcc, html
from web.components.navigation import create_navigation, create_page_help

# Import the new sections
from web.apps.prediction_parameters import get_parameters_layout
from web.apps.prediction_results import get_results_layout
from web.apps.prediction_visualization import get_visualization_layout

# Import the callbacks package to register all modularized callbacks
import web.apps.prediction_callbacks  # noqa: F401

# Styles communs
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '24px'
}

help_text = """
### Prédiction (Deep Learning)

Cette page est le cœur du système d'intelligence artificielle. Elle permet de créer, entraîner et évaluer des modèles de prédiction sur des données réelles.

#### 1. Paramètres du Modèle
C'est ici que vous définissez la "recette" de votre IA.
*   **Actions** : Sélectionnez les actifs sur lesquels le modèle va apprendre (ex: AAPL, MSFT).
*   **Période** : Intervalle de temps historique utilisé pour l'entraînement (ex: 2020-2023).
*   **Architecture** : Type de réseau de neurones (LSTM, GRU, Transformers...).
*   **Hyperparamètres** :
    *   `Learning Rate` : Vitesse d'apprentissage.
    *   `Epochs` : Durée de l'entraînement.
    *   `Layers` : Profondeur du réseau.

#### 2. Résultats de l'entraînement
Une fois l'entraînement lancé, cette section affiche les performances en temps réel.
*   **Courbe de Loss** : Montre l'erreur du modèle au fil du temps. Elle doit descendre et se stabiliser.
*   **Métriques** : Accuracy, MAE (Mean Absolute Error), etc.

#### 3. Visualisation
Permet de vérifier visuellement la qualité des prédictions.
*   **Comparaison** : Superpose la courbe de prix réelle et la courbe prédite par l'IA.
*   **Test Set** : L'évaluation se fait sur des données que le modèle n'a *jamais vues* pendant l'entraînement, pour garantir qu'il ne triche pas (pas de par cœur).
"""

layout = html.Div([
    create_page_help("Aide Prédiction", help_text),
    
    # En-tête fixe
    html.Div([
        html.H3('Prédiction', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Intelligence Artificielle & Deep Learning', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0',
            'fontSize': '0.9375rem'
        })
    ], style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'right': 0,
        'backgroundColor': '#0a0a0f',
        'padding': '20px 24px',
        'zIndex': 1000,
        'borderBottom': '1px solid rgba(148, 163, 184, 0.1)',
        'backdropFilter': 'blur(10px)'
    }),

    # Contenu scrollable
    html.Div([
        # Section Paramètres
        html.Div([
            html.Div([
                html.Span('⚙️', style={'fontSize': '1.25rem'}),
                html.Span('Paramètres du Modèle', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            get_parameters_layout(),
        ], style=CARD_STYLE),

        # Section Résultats
        html.Div([
            html.Div([
                html.Span('📊', style={'fontSize': '1.25rem'}),
                html.Span('Résultats du Modèle', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            get_results_layout(),
        ], style=CARD_STYLE),

        # Section Visualisation
        html.Div([
            html.Div([
                html.Span('📈', style={'fontSize': '1.25rem'}),
                html.Span('Visualisation des Prédictions', style={
                    'fontSize': '1.125rem',
                    'fontWeight': '600',
                    'color': '#a78bfa',
                    'marginLeft': '10px'
                })
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            get_visualization_layout(),
        ], style=CARD_STYLE),

        # Spacer pour navigation
        html.Div(style={'height': '100px'}),

        create_navigation()
    ], style={
        'marginTop': '100px',
        'padding': '24px 32px',
        'backgroundColor': '#0a0a0f',
        'minHeight': 'calc(100vh - 100px)',
        'width': '100%',
        'maxWidth': '100%',
        'margin': '100px 0 0'
    })
], style={
    'backgroundColor': '#0a0a0f',
    'minHeight': '100vh',
    'width': '100%'
})
