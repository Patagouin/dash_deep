"""
Configuration centralisée des modèles IA et de leurs paramètres.
Ce fichier factorise les constantes, icônes et options pour le Playground et la Prédiction.
"""

from dash import dcc, html

# ==============================================================================
# Constantes par défaut (évite les magic numbers)
# ==============================================================================

# Paramètres de données
DEFAULT_LOOK_BACK = 60
DEFAULT_STRIDE = 1
DEFAULT_NB_Y = 5
DEFAULT_FIRST_MINUTES = 60

# Paramètres d'entraînement
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRAIN_TEST_RATIO = 70  # en %

# Paramètres LSTM
DEFAULT_LSTM_UNITS = 64
DEFAULT_LSTM_LAYERS = 1

# Paramètres Transformer
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_TRANSFORMER_LAYERS = 2
DEFAULT_FF_MULTIPLIER = 4
DEFAULT_DROPOUT = 0.1

# Paramètres Hybride
DEFAULT_FUSION_MODE = 'concat'
DEFAULT_HYBRID_LSTM_UNITS = 64
DEFAULT_HYBRID_LSTM_LAYERS = 1
DEFAULT_HYBRID_EMBED_DIM = 64
DEFAULT_HYBRID_NUM_HEADS = 4
DEFAULT_HYBRID_TRANS_LAYERS = 1

# Paramètres financiers
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_TRADE_AMOUNT = 1_000.0
DEFAULT_K_TRADES = 2
DEFAULT_SPREAD_PCT = 0.0

# Divers
TRAINING_GRAPH_UPDATE_INTERVAL_SECONDS = 5.0


# ==============================================================================
# Icônes et labels des types de modèles
# ==============================================================================

MODEL_TYPES = {
    'lstm': {
        'icon': '🔄',
        'label': 'LSTM (Récurrent)',
        'short_label': 'LSTM',
        'color': '#1f77b4',
        'description': 'Réseau récurrent Long Short-Term Memory. Excellent pour les dépendances séquentielles locales.'
    },
    'gru': {
        'icon': '🔃',
        'label': 'GRU (Récurrent)',
        'short_label': 'GRU',
        'color': '#17becf',
        'description': 'Gated Recurrent Unit. Variante simplifiée du LSTM, plus rapide à entraîner.'
    },
    'transformer': {
        'icon': '🎯',
        'label': 'Transformer (Attention)',
        'short_label': 'Transformer',
        'color': '#2ca02c',
        'description': 'Architecture avec attention multi-têtes. Voit toutes les relations dans la séquence d\'un coup.'
    },
    'hybrid': {
        'icon': '🔀',
        'label': 'Hybride LSTM + Transformer',
        'short_label': 'Hybride',
        'color': '#9467bd',
        'description': 'Combine la mémoire séquentielle du LSTM avec la vision globale du Transformer.'
    }
}

# Modes de fusion pour le modèle Hybride
FUSION_MODES = {
    'concat': {
        'label': 'Concat',
        'description': 'Concaténation simple des deux vecteurs [LSTM | Transformer]'
    },
    'add': {
        'label': 'Add',
        'description': 'Addition des représentations (après projection)'
    },
    'attention': {
        'label': 'Attention',
        'description': 'Le LSTM "interroge" le Transformer via cross-attention'
    }
}


# ==============================================================================
# Tooltips (infobulles) communs
# ==============================================================================

TOOLTIPS = {
    # Paramètres de données
    'look_back': 'Taille de la fenêtre d\'entrée (en points/minutes)',
    'stride': 'Pas d\'échantillonnage pour la fenêtre d\'entrée (ex: 5 = 1 point toutes les 5 min)',
    'nb_y': 'Nombre de points futurs à prédire',
    'first_minutes': 'Nombre de minutes d\'observation en début de journée (Input du modèle)',
    'prediction_type': 'Type de cible à prédire : Variation (Return) ou Prix Normalisé (Price)',
    'directional_accuracy': 'Activer la métrique Directional Accuracy (pourcentage de bonnes directions)',
    
    # Paramètres d'entraînement
    'learning_rate': 'Vitesse d\'apprentissage (Learning Rate)',
    'epochs': 'Nombre d\'itérations complètes sur le jeu d\'entraînement',
    'train_test_ratio': 'Ratio Entraînement/Test en pourcentage',
    
    # LSTM
    'lstm_units': 'Nombre de neurones par couche LSTM',
    'lstm_layers': 'Nombre de couches LSTM empilées',
    
    # Transformer
    'model_type': 'Type de modèle IA : LSTM classique, Transformer avec attention, ou Hybride LSTM+Transformer',
    'embed_dim': 'Dimension des embeddings internes du Transformer',
    'num_heads': 'Nombre de têtes d\'attention (parallélise l\'attention sur différents aspects)',
    'transformer_layers': 'Nombre de blocs Transformer empilés',
    'ff_multiplier': 'Multiplicateur pour la couche Feed-Forward (ff_dim = embed_dim × multiplier)',
    'dropout': 'Taux de dropout pour la régularisation (prévient le surapprentissage)',
    
    # Hybride
    'fusion_mode': 'Mode de fusion pour le modèle Hybride : concat, add, ou attention croisée',
    
    # Financier
    'initial_cash': 'Capital de départ pour la simulation',
    'trade_amount': 'Montant engagé par trade',
    'k_trades': 'Nombre maximum de trades simultanés/journaliers',
    'spread': 'Spread bid-ask en % appliqué à chaque trade (coût de transaction)',
}


# ==============================================================================
# Fonctions utilitaires pour générer les composants UI
# ==============================================================================

def get_model_type_options(include_gru: bool = True, include_hybrid: bool = True):
    """
    Génère les options pour le dropdown de type de modèle avec icônes.
    
    Args:
        include_gru: Inclure le type GRU
        include_hybrid: Inclure le type Hybride
    
    Returns:
        Liste d'options pour dcc.Dropdown
    """
    options = [
        {
            'label': f"{MODEL_TYPES['lstm']['icon']} {MODEL_TYPES['lstm']['label']}",
            'value': 'lstm'
        }
    ]
    
    if include_gru:
        options.append({
            'label': f"{MODEL_TYPES['gru']['icon']} {MODEL_TYPES['gru']['label']}",
            'value': 'gru'
        })
    
    options.append({
        'label': f"{MODEL_TYPES['transformer']['icon']} {MODEL_TYPES['transformer']['label']}",
        'value': 'transformer'
    })
    
    if include_hybrid:
        options.append({
            'label': f"{MODEL_TYPES['hybrid']['icon']} {MODEL_TYPES['hybrid']['label']}",
            'value': 'hybrid'
        })
    
    return options


def get_fusion_mode_options():
    """
    Génère les options pour le dropdown de mode de fusion (Hybride).
    
    Returns:
        Liste d'options pour dcc.Dropdown
    """
    return [
        {'label': info['label'], 'value': mode}
        for mode, info in FUSION_MODES.items()
    ]


def get_model_icon(model_type: str) -> str:
    """Retourne l'icône correspondant au type de modèle."""
    return MODEL_TYPES.get(model_type, MODEL_TYPES['lstm'])['icon']


def get_model_color(model_type: str) -> str:
    """Retourne la couleur correspondant au type de modèle."""
    return MODEL_TYPES.get(model_type, MODEL_TYPES['lstm'])['color']


# ==============================================================================
# Composants UI réutilisables pour les paramètres de modèle
# ==============================================================================

def create_lstm_params_section(id_prefix: str = '', persistence: bool = True):
    """
    Crée la section des paramètres LSTM.
    
    Args:
        id_prefix: Préfixe pour les IDs (ex: 'play_' pour Playground)
        persistence: Activer la persistence en session
    
    Returns:
        html.Div contenant les paramètres LSTM
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    return html.Div([
        html.Div([
            html.Label('Unités LSTM', title=TOOLTIPS['lstm_units']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}lstm_units',
                    value=DEFAULT_LSTM_UNITS,
                    type='number',
                    step=8,
                    min=8,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['lstm_units']
            ),
        ]),
        html.Div([
            html.Label('Couches LSTM', title=TOOLTIPS['lstm_layers']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}lstm_layers',
                    value=DEFAULT_LSTM_LAYERS,
                    type='number',
                    step=1,
                    min=1,
                    max=4,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['lstm_layers']
            ),
        ]),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))',
        'gap': '8px'
    })


def create_transformer_params_section(id_prefix: str = '', persistence: bool = True):
    """
    Crée la section des paramètres Transformer.
    
    Args:
        id_prefix: Préfixe pour les IDs (ex: 'play_' pour Playground)
        persistence: Activer la persistence en session
    
    Returns:
        html.Div contenant les paramètres Transformer
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    return html.Div([
        html.Div([
            html.Label('Embed dim', title=TOOLTIPS['embed_dim']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}embed_dim',
                    value=DEFAULT_EMBED_DIM,
                    type='number',
                    step=8,
                    min=16,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['embed_dim']
            ),
        ]),
        html.Div([
            html.Label('Num heads', title=TOOLTIPS['num_heads']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}num_heads',
                    value=DEFAULT_NUM_HEADS,
                    type='number',
                    step=1,
                    min=1,
                    max=16,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['num_heads']
            ),
        ]),
        html.Div([
            html.Label('Transformer layers', title=TOOLTIPS['transformer_layers']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}transformer_layers',
                    value=DEFAULT_TRANSFORMER_LAYERS,
                    type='number',
                    step=1,
                    min=1,
                    max=6,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['transformer_layers']
            ),
        ]),
        html.Div([
            html.Label('FF multiplier', title=TOOLTIPS['ff_multiplier']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}ff_multiplier',
                    value=DEFAULT_FF_MULTIPLIER,
                    type='number',
                    step=1,
                    min=1,
                    max=8,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['ff_multiplier']
            ),
        ]),
        html.Div([
            html.Label('Dropout', title=TOOLTIPS['dropout']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}dropout',
                    value=DEFAULT_DROPOUT,
                    type='number',
                    step=0.05,
                    min=0.0,
                    max=0.5,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['dropout']
            ),
        ]),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))',
        'gap': '8px'
    })


def create_hybrid_params_section(id_prefix: str = '', persistence: bool = True):
    """
    Crée la section des paramètres Hybride (LSTM + Transformer).
    
    Args:
        id_prefix: Préfixe pour les IDs (ex: 'play_' pour Playground)
        persistence: Activer la persistence en session
    
    Returns:
        html.Div contenant les paramètres Hybride
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    return html.Div([
        html.Div([
            html.Label('LSTM units', title=TOOLTIPS['lstm_units']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_lstm_units',
                    value=DEFAULT_HYBRID_LSTM_UNITS,
                    type='number',
                    step=8,
                    min=8,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['lstm_units']
            ),
        ]),
        html.Div([
            html.Label('LSTM layers', title=TOOLTIPS['lstm_layers']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_lstm_layers',
                    value=DEFAULT_HYBRID_LSTM_LAYERS,
                    type='number',
                    step=1,
                    min=1,
                    max=3,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['lstm_layers']
            ),
        ]),
        html.Div([
            html.Label('Embed dim', title=TOOLTIPS['embed_dim']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_embed_dim',
                    value=DEFAULT_HYBRID_EMBED_DIM,
                    type='number',
                    step=8,
                    min=16,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['embed_dim']
            ),
        ]),
        html.Div([
            html.Label('Trans. heads', title=TOOLTIPS['num_heads']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_num_heads',
                    value=DEFAULT_HYBRID_NUM_HEADS,
                    type='number',
                    step=1,
                    min=1,
                    max=8,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['num_heads']
            ),
        ]),
        html.Div([
            html.Label('Trans. layers', title=TOOLTIPS['transformer_layers']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_trans_layers',
                    value=DEFAULT_HYBRID_TRANS_LAYERS,
                    type='number',
                    step=1,
                    min=1,
                    max=4,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['transformer_layers']
            ),
        ]),
        html.Div([
            html.Label('Fusion mode', title=TOOLTIPS['fusion_mode']),
            dcc.Dropdown(
                id=f'{id_prefix}fusion_mode',
                options=get_fusion_mode_options(),
                value=DEFAULT_FUSION_MODE,
                style={'width': '100%', 'color': '#FF8C00'},
                **persistence_props
            ),
        ]),
        html.Div([
            html.Label('Dropout', title=TOOLTIPS['dropout']),
            html.Div(
                dcc.Input(
                    id=f'{id_prefix}hybrid_dropout',
                    value=DEFAULT_DROPOUT,
                    type='number',
                    step=0.05,
                    min=0.0,
                    max=0.5,
                    style={'width': '100%'},
                    **persistence_props
                ),
                title=TOOLTIPS['dropout']
            ),
        ]),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))',
        'gap': '8px'
    })


# ==============================================================================
# Styles CSS réutilisables
# ==============================================================================

STYLES = {
    'grid_params': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))',
        'gap': '8px'
    },
    'grid_params_small': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))',
        'gap': '8px'
    },
    'section_label': {
        'fontWeight': 'bold',
        'marginBottom': '4px',
        'marginTop': '12px'
    },
    'hidden': {
        'display': 'none'
    },
    'show_grid': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))',
        'gap': '8px'
    },
    'show_block': {
        'display': 'block'
    }
}


def get_label_style(model_type: str):
    """Retourne le style du label pour un type de modèle."""
    color = get_model_color(model_type)
    return {
        'fontWeight': 'bold',
        'color': color,
        'marginBottom': '4px',
        'marginTop': '12px'
    }

