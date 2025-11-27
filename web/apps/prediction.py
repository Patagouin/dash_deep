from dash import dcc, html
from web.components.navigation import create_navigation, create_page_help

# Import the new sections
from web.apps.prediction_parameters import get_parameters_layout
from web.apps.prediction_results import get_results_layout
from web.apps.prediction_visualization import get_visualization_layout

# Import the callbacks package to register all modularized callbacks
import web.apps.prediction_callbacks  # noqa: F401

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
