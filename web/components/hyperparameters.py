# -*- coding: utf-8 -*-
"""
Composants UI pour les hyperparamètres ML.
Réutilisables dans playground.py, prediction.py, etc.
"""

from dash import dcc, html


# ==============================================================================
# TOOLTIPS CENTRALISÉS
# ==============================================================================

TOOLTIPS = {
    # Paramètres de données
    'look_back': "Taille de la fenêtre d'entrée (en points/minutes)",
    'stride': "Pas d'échantillonnage pour la fenêtre d'entrée (ex: 5 = 1 point toutes les 5 min)",
    'nb_y': "Nombre de points futurs à prédire (répartis uniformément sur le reste de la journée)",
    'first_minutes': "Nombre de minutes d'observation en début de journée (Input du modèle)",
    'prediction_type': "Type de cible à prédire : Variation (Return) ou Prix Normalisé (Price)",
    'directional_accuracy': "Activer la métrique Directional Accuracy (pourcentage de bonnes directions)",
    'loss_type': '''Type de fonction de perte (Loss) pour l'entraînement:
• MSE (défaut): Mean Squared Error - erreur quadratique moyenne.
• Scaled MSE (×100): MSE multiplié par 100 - loss plus lisible.
• MAE: Mean Absolute Error - erreur absolue moyenne.''',
    
    # Paramètres LSTM
    'lstm_units': "Nombre de neurones par couche LSTM",
    'lstm_layers': "Nombre de couches LSTM empilées",
    
    # Paramètres Transformer
    'embed_dim': "Dimension des embeddings (vecteurs internes). Plus grand = plus expressif mais plus lent.",
    'num_heads': "Nombre de têtes d'attention. embed_dim doit être divisible par num_heads.",
    'transformer_layers': "Nombre de couches Transformer empilées.",
    'ff_multiplier': "Multiplicateur pour la couche Feed-Forward (généralement 4×embed_dim).",
    'dropout': "Taux de dropout pour régularisation (0.0 = pas de dropout, 0.5 = 50% des connexions).",
    
    # Paramètres hybride
    'fusion_mode': '''Mode de fusion LSTM + Transformer:
• Concat: Concatène les deux représentations [LSTM | Transformer]
• Add: Additionne après projection (plus compact)
• Attention: Le LSTM "interroge" le Transformer via cross-attention (le plus expressif)''',
    
    # Paramètres d'entraînement
    'learning_rate': "Vitesse d'apprentissage (Learning Rate)",
    'epochs': "Nombre d'itérations complètes sur le jeu d'entraînement",
    
    # Paramètres de trading
    'initial_cash': "Capital de départ pour la simulation",
    'trade_amount': "Montant engagé par trade",
    'k_trades': "Nombre maximum de trades simultanés/journaliers",
    'spread': "Spread bid-ask en % appliqué à chaque trade (coût de transaction)",
}


# ==============================================================================
# VALEURS PAR DÉFAUT
# ==============================================================================

DEFAULTS = {
    # Données
    'look_back': 60,
    'stride': 1,
    'nb_y': 5,
    'first_minutes': 60,
    
    # LSTM
    'lstm_units': 64,
    'lstm_layers': 1,
    
    # Transformer
    'embed_dim': 64,
    'num_heads': 4,
    'transformer_layers': 2,
    'ff_multiplier': 4,
    'dropout': 0.1,
    
    # Hybride
    'fusion_mode': 'concat',
    
    # Entraînement
    'learning_rate': 0.001,
    'epochs': 5,
    
    # Trading
    'initial_cash': 10000,
    'trade_amount': 1000,
    'k_trades': 2,
    'spread': 0.0,
}


# ==============================================================================
# COMPOSANTS UI RÉUTILISABLES
# ==============================================================================

def create_labeled_input(
    label: str,
    id: str,
    value=None,
    type: str = 'number',
    tooltip: str = None,
    step=None,
    min_val=None,
    max_val=None,
    persistence: bool = True,
    style: dict = None
):
    """
    Crée un input avec label et tooltip optionnel.
    
    Args:
        label: Texte du label
        id: ID du composant Dash
        value: Valeur par défaut
        type: Type d'input ('number', 'text')
        tooltip: Texte d'info-bulle (utilise TOOLTIPS si None et clé existe)
        step: Pas pour les inputs numériques
        min_val: Valeur minimum
        max_val: Valeur maximum
        persistence: Activer la persistance session
        style: Style CSS additionnel
    """
    # Récupérer tooltip depuis TOOLTIPS si non fourni
    tooltip_text = tooltip or TOOLTIPS.get(id.split('_')[-1], '') or TOOLTIPS.get(id, '')
    
    input_style = {'width': '100%'}
    if style:
        input_style.update(style)
    
    input_props = {
        'id': id,
        'value': value,
        'type': type,
        'style': input_style,
        'persistence': persistence,
        'persistence_type': 'session',
    }
    
    if step is not None:
        input_props['step'] = step
    if min_val is not None:
        input_props['min'] = min_val
    if max_val is not None:
        input_props['max'] = max_val
    
    return html.Div([
        html.Label(label, title=tooltip_text),
        html.Div(
            dcc.Input(**input_props),
            title=tooltip_text
        )
    ])


def create_labeled_dropdown(
    label: str,
    id: str,
    options: list,
    value=None,
    tooltip: str = None,
    multi: bool = False,
    persistence: bool = True,
    style: dict = None
):
    """
    Crée un dropdown avec label et tooltip optionnel.
    
    Args:
        label: Texte du label
        id: ID du composant Dash
        options: Liste d'options [{'label': ..., 'value': ...}, ...]
        value: Valeur par défaut
        tooltip: Texte d'info-bulle
        multi: Autoriser la sélection multiple
        persistence: Activer la persistance session
        style: Style CSS additionnel
    """
    tooltip_text = tooltip or TOOLTIPS.get(id, '')
    
    dropdown_style = {'width': '100%', 'color': '#FF8C00'}
    if style:
        dropdown_style.update(style)
    
    return html.Div([
        html.Label(label, title=tooltip_text),
        dcc.Dropdown(
            id=id,
            options=options,
            value=value,
            multi=multi,
            persistence=persistence,
            persistence_type='session',
            style=dropdown_style
        )
    ], title=tooltip_text)


def create_labeled_slider(
    label: str,
    id: str,
    min_val: int,
    max_val: int,
    value: int = None,
    step: int = 1,
    marks: dict = None,
    tooltip: str = None,
    persistence: bool = True
):
    """
    Crée un slider avec label et tooltip optionnel.
    """
    tooltip_text = tooltip or TOOLTIPS.get(id, '')
    
    if marks is None:
        marks = {min_val: str(min_val), max_val: str(max_val)}
    
    return html.Div([
        html.Label(label, title=tooltip_text),
        html.Div([
            dcc.Slider(
                id=id,
                min=min_val,
                max=max_val,
                step=step,
                value=value or min_val,
                marks=marks,
                persistence=persistence,
                persistence_type='session'
            )
        ], title=tooltip_text)
    ])


# ==============================================================================
# PANNEAUX DE PARAMÈTRES COMPLETS
# ==============================================================================

def create_lstm_params_panel(id_prefix: str = 'play', show: bool = True):
    """
    Crée le panneau de paramètres LSTM.
    
    Args:
        id_prefix: Préfixe pour les IDs (ex: 'play' → 'play_units', 'play_layers')
        show: Afficher ou masquer le panneau
    """
    display = 'grid' if show else 'none'
    
    return html.Div([
        html.Div([
            html.Label('🔄 Architecture LSTM', style={
                'fontWeight': 'bold',
                'color': '#1f77b4',
                'marginBottom': '4px',
                'marginTop': '12px'
            }),
        ], id=f'label_{id_prefix}_lstm_params'),
        html.Div([
            create_labeled_input(
                'Unités LSTM',
                f'{id_prefix}_units',
                DEFAULTS['lstm_units'],
                step=1,
                min_val=4,
                tooltip=TOOLTIPS['lstm_units']
            ),
            create_labeled_input(
                'Couches LSTM',
                f'{id_prefix}_layers',
                DEFAULTS['lstm_layers'],
                step=1,
                min_val=1,
                max_val=4,
                tooltip=TOOLTIPS['lstm_layers']
            ),
        ], id=f'panel_{id_prefix}_lstm_params', style={
            'display': display,
            'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))',
            'gap': '8px'
        })
    ])


def create_transformer_params_panel(id_prefix: str = 'play', show: bool = False):
    """
    Crée le panneau de paramètres Transformer.
    """
    display = 'grid' if show else 'none'
    
    return html.Div([
        html.Div([
            html.Label('🎯 Architecture Transformer', style={
                'fontWeight': 'bold',
                'color': '#2ca02c',
                'marginBottom': '4px',
                'marginTop': '12px'
            }),
        ], id=f'label_{id_prefix}_transformer_params', style={'display': 'block' if show else 'none'}),
        html.Div([
            create_labeled_input(
                'Embed dim',
                f'{id_prefix}_embed_dim',
                DEFAULTS['embed_dim'],
                step=8,
                min_val=16,
                tooltip=TOOLTIPS['embed_dim']
            ),
            create_labeled_input(
                'Num heads',
                f'{id_prefix}_num_heads',
                DEFAULTS['num_heads'],
                step=1,
                min_val=1,
                max_val=16,
                tooltip=TOOLTIPS['num_heads']
            ),
            create_labeled_input(
                'Transformer layers',
                f'{id_prefix}_transformer_layers',
                DEFAULTS['transformer_layers'],
                step=1,
                min_val=1,
                max_val=6,
                tooltip=TOOLTIPS['transformer_layers']
            ),
            create_labeled_input(
                'FF multiplier',
                f'{id_prefix}_ff_multiplier',
                DEFAULTS['ff_multiplier'],
                step=1,
                min_val=1,
                max_val=8,
                tooltip=TOOLTIPS['ff_multiplier']
            ),
            create_labeled_input(
                'Dropout',
                f'{id_prefix}_dropout',
                DEFAULTS['dropout'],
                step=0.05,
                min_val=0.0,
                max_val=0.5,
                tooltip=TOOLTIPS['dropout']
            ),
        ], id=f'panel_{id_prefix}_transformer_params', style={
            'display': display,
            'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))',
            'gap': '8px'
        })
    ])


def create_hybrid_params_panel(id_prefix: str = 'play', show: bool = False):
    """
    Crée le panneau de paramètres Hybride (LSTM + Transformer).
    """
    display = 'grid' if show else 'none'
    
    fusion_options = [
        {'label': '🔗 Concat', 'value': 'concat'},
        {'label': '➕ Add', 'value': 'add'},
        {'label': '🎯 Attention', 'value': 'attention'},
    ]
    
    return html.Div([
        html.Div([
            html.Label('🔀 Architecture Hybride', style={
                'fontWeight': 'bold',
                'color': '#9467bd',
                'marginBottom': '4px',
                'marginTop': '12px'
            }),
        ], id=f'label_{id_prefix}_hybrid_params', style={'display': 'block' if show else 'none'}),
        html.Div([
            create_labeled_input(
                'LSTM units',
                f'{id_prefix}_hybrid_lstm_units',
                DEFAULTS['lstm_units'],
                step=8,
                min_val=8,
                tooltip=TOOLTIPS['lstm_units']
            ),
            create_labeled_input(
                'LSTM layers',
                f'{id_prefix}_hybrid_lstm_layers',
                DEFAULTS['lstm_layers'],
                step=1,
                min_val=1,
                max_val=3,
                tooltip=TOOLTIPS['lstm_layers']
            ),
            create_labeled_input(
                'Embed dim',
                f'{id_prefix}_hybrid_embed_dim',
                DEFAULTS['embed_dim'],
                step=8,
                min_val=16,
                tooltip=TOOLTIPS['embed_dim']
            ),
            create_labeled_input(
                'Trans. heads',
                f'{id_prefix}_hybrid_num_heads',
                DEFAULTS['num_heads'],
                step=1,
                min_val=1,
                max_val=8,
                tooltip=TOOLTIPS['num_heads']
            ),
            create_labeled_input(
                'Trans. layers',
                f'{id_prefix}_hybrid_trans_layers',
                1,
                step=1,
                min_val=1,
                max_val=4,
                tooltip=TOOLTIPS['transformer_layers']
            ),
            create_labeled_dropdown(
                'Fusion mode',
                f'{id_prefix}_fusion_mode',
                fusion_options,
                DEFAULTS['fusion_mode'],
                tooltip=TOOLTIPS['fusion_mode']
            ),
            create_labeled_input(
                'Dropout',
                f'{id_prefix}_hybrid_dropout',
                DEFAULTS['dropout'],
                step=0.05,
                min_val=0.0,
                max_val=0.5,
                tooltip=TOOLTIPS['dropout']
            ),
        ], id=f'panel_{id_prefix}_hybrid_params', style={
            'display': display,
            'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))',
            'gap': '8px'
        })
    ])


def create_training_params_panel(id_prefix: str = 'play'):
    """
    Crée le panneau de paramètres d'entraînement (learning rate, epochs).
    """
    return html.Div([
        html.Div([
            html.Label('⚙️ Entraînement', style={
                'fontWeight': 'bold',
                'color': '#FF8C00',
                'marginBottom': '4px',
                'marginTop': '12px'
            }),
        ]),
        html.Div([
            create_labeled_input(
                'Learning rate',
                f'{id_prefix}_lr',
                DEFAULTS['learning_rate'],
                step=0.0001,
                tooltip=TOOLTIPS['learning_rate']
            ),
            create_labeled_input(
                'Epochs',
                f'{id_prefix}_epochs',
                DEFAULTS['epochs'],
                step=1,
                min_val=1,
                tooltip=TOOLTIPS['epochs']
            ),
        ], id=f'panel_{id_prefix}_training', style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))',
            'gap': '8px'
        })
    ])


def create_backtest_params_panel(id_prefix: str = 'play'):
    """
    Crée le panneau de paramètres de backtest (capital, trades, spread).
    """
    strategy_options = [
        {'label': '📈 LONG (hausse)', 'value': 'long'},
        {'label': '📉 SHORT (baisse)', 'value': 'short'},
        {'label': '📊 LONG & SHORT', 'value': 'both'},
    ]
    
    return html.Div([
        html.Div([
            html.Label('💰 Simulation Financière (Backtest)', style={
                'fontWeight': 'bold',
                'color': '#FF8C00',
                'marginBottom': '4px'
            }),
        ]),
        html.Div([
            create_labeled_input(
                'Capital initial (€)',
                f'{id_prefix}_initial_cash',
                DEFAULTS['initial_cash'],
                step=100,
                min_val=0,
                tooltip=TOOLTIPS['initial_cash']
            ),
            create_labeled_input(
                'Montant par trade (€)',
                f'{id_prefix}_trade_amount',
                DEFAULTS['trade_amount'],
                step=50,
                min_val=0,
                tooltip=TOOLTIPS['trade_amount']
            ),
            create_labeled_input(
                'K trades/jour',
                f'{id_prefix}_k_trades',
                DEFAULTS['k_trades'],
                step=1,
                min_val=1,
                max_val=10,
                tooltip=TOOLTIPS['k_trades']
            ),
            create_labeled_input(
                'Spread (%)',
                f'{id_prefix}_spread_pct',
                DEFAULTS['spread'],
                step=0.01,
                min_val=0.0,
                max_val=1.0,
                tooltip=TOOLTIPS['spread']
            ),
            create_labeled_dropdown(
                'Stratégie',
                f'{id_prefix}_strategy',
                strategy_options,
                'long',
                tooltip='LONG = acheter puis vendre (gagner si hausse). SHORT = vendre puis racheter (gagner si baisse).'
            ),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))',
            'gap': '8px'
        })
    ])


# ==============================================================================
# UTILITAIRES
# ==============================================================================

def get_model_type_options(include_gru: bool = False, include_hybrid: bool = True):
    """
    Retourne les options de type de modèle pour un dropdown.
    """
    options = [
        {'label': '🔄 LSTM', 'value': 'lstm'},
    ]
    
    if include_gru:
        options.append({'label': '🔁 GRU', 'value': 'gru'})
    
    options.append({'label': '🎯 Transformer', 'value': 'transformer'})
    
    if include_hybrid:
        options.append({'label': '🔀 Hybride LSTM+Trans', 'value': 'hybrid'})
    
    return options


def get_loss_type_options():
    """
    Retourne les options de type de loss pour un dropdown.
    """
    return [
        {'label': 'MSE (défaut)', 'value': 'mse'},
        {'label': 'Scaled MSE (×100)', 'value': 'scaled_mse'},
        {'label': 'MAE', 'value': 'mae'},
    ]


def get_prediction_type_options():
    """
    Retourne les options de type de prédiction pour un RadioItems.
    """
    return [
        {'label': 'Variation (%)', 'value': 'return'},
        {'label': 'Prix', 'value': 'price'},
        {'label': 'Signal / Index', 'value': 'signal'},
    ]

