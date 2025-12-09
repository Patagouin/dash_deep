"""
Composants UI pour les hyperparamètres.
Utilisé par prediction et playground pour créer des contrôles de paramètres cohérents.
"""

from dash import dcc, html
from typing import Dict, Any, List, Optional


# ==============================================================================
# Constantes par défaut
# ==============================================================================

DEFAULT_LOOK_BACK = 60
DEFAULT_STRIDE = 1
DEFAULT_NB_Y = 5
DEFAULT_FIRST_MINUTES = 60
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_LSTM_UNITS = 64
DEFAULT_LSTM_LAYERS = 1
DEFAULT_EMBED_DIM = 64
DEFAULT_NUM_HEADS = 4
DEFAULT_TRANSFORMER_LAYERS = 2
DEFAULT_FF_MULTIPLIER = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_FUSION_MODE = 'concat'
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_TRADE_AMOUNT = 1_000.0
DEFAULT_K_TRADES = 2


# ==============================================================================
# Types de modèles avec icônes
# ==============================================================================

MODEL_TYPES = {
    'lstm': {
        'icon': '🔄',
        'label': 'LSTM (Récurrent)',
        'short_label': 'LSTM',
        'color': '#1f77b4',
    },
    'gru': {
        'icon': '🔃',
        'label': 'GRU (Récurrent)',
        'short_label': 'GRU',
        'color': '#17becf',
    },
    'transformer': {
        'icon': '🎯',
        'label': 'Transformer (Attention)',
        'short_label': 'Transformer',
        'color': '#2ca02c',
    },
    'hybrid': {
        'icon': '🔀',
        'label': 'Hybride LSTM + Transformer',
        'short_label': 'Hybride',
        'color': '#9467bd',
    }
}

FUSION_MODES = {
    'concat': {'label': 'Concat'},
    'add': {'label': 'Add'},
    'attention': {'label': 'Attention'},
}


# ==============================================================================
# Tooltips communs
# ==============================================================================

TOOLTIPS = {
    'look_back': 'Taille de la fenêtre d\'entrée (en points/minutes)',
    'stride': 'Pas d\'échantillonnage pour la fenêtre d\'entrée',
    'nb_y': 'Nombre de points futurs à prédire',
    'first_minutes': 'Nombre de minutes d\'observation en début de journée',
    'learning_rate': 'Vitesse d\'apprentissage (Learning Rate)',
    'epochs': 'Nombre d\'itérations complètes sur le jeu d\'entraînement',
    'lstm_units': 'Nombre de neurones par couche LSTM',
    'lstm_layers': 'Nombre de couches LSTM empilées',
    'model_type': 'Type de modèle IA',
    'embed_dim': 'Dimension des embeddings internes du Transformer',
    'num_heads': 'Nombre de têtes d\'attention',
    'transformer_layers': 'Nombre de blocs Transformer empilés',
    'ff_multiplier': 'Multiplicateur pour la couche Feed-Forward',
    'dropout': 'Taux de dropout pour la régularisation',
    'fusion_mode': 'Mode de fusion pour le modèle Hybride',
    'initial_cash': 'Capital de départ pour la simulation',
    'trade_amount': 'Montant engagé par trade',
    'k_trades': 'Nombre maximum de trades simultanés/journaliers',
}


# ==============================================================================
# Fonctions utilitaires
# ==============================================================================

def get_model_type_options(include_gru: bool = True, include_hybrid: bool = True) -> List[Dict]:
    """Génère les options pour le dropdown de type de modèle."""
    options = [
        {'label': f"{MODEL_TYPES['lstm']['icon']} {MODEL_TYPES['lstm']['label']}", 'value': 'lstm'}
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


def get_fusion_mode_options() -> List[Dict]:
    """Génère les options pour le dropdown de mode de fusion."""
    return [{'label': info['label'], 'value': mode} for mode, info in FUSION_MODES.items()]


def get_model_icon(model_type: str) -> str:
    """Retourne l'icône correspondant au type de modèle."""
    return MODEL_TYPES.get(model_type, MODEL_TYPES['lstm'])['icon']


def get_model_color(model_type: str) -> str:
    """Retourne la couleur correspondant au type de modèle."""
    return MODEL_TYPES.get(model_type, MODEL_TYPES['lstm'])['color']


# ==============================================================================
# Composants UI réutilisables
# ==============================================================================

def create_number_input(
    id: str,
    label: str,
    default: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    step: float = 1,
    tooltip: str = '',
    with_save_button: bool = False,
    save_button_id: Optional[str] = None,
    persistence: bool = True
) -> html.Div:
    """
    Crée un input numérique avec label et tooltip optionnel.
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    input_component = dcc.Input(
        id=id,
        type='number',
        value=default,
        min=min_val,
        max=max_val,
        step=step,
        style={'width': '100%', 'color': '#FF8C00'},
        **persistence_props
    )
    
    if with_save_button and save_button_id:
        input_row = html.Div([
            input_component,
            html.Button('+', id=save_button_id, n_clicks=0, style={'marginLeft': '10px'})
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'minmax(0, 1fr) auto',
            'alignItems': 'center',
            'gap': '10px'
        })
    else:
        input_row = html.Div(input_component, title=tooltip)
    
    return html.Div([
        html.Label(label, style={'paddingLeft': '10px'}, title=tooltip),
        input_row
    ])


def create_dropdown_input(
    id: str,
    label: str,
    options: List[Dict],
    default: Any,
    tooltip: str = '',
    with_save_button: bool = False,
    save_button_id: Optional[str] = None,
    persistence: bool = True,
    multi: bool = False
) -> html.Div:
    """
    Crée un dropdown avec label et tooltip optionnel.
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    dropdown = dcc.Dropdown(
        id=id,
        options=options,
        value=default,
        multi=multi,
        style={'width': '100%', 'color': '#FF8C00'},
        **persistence_props
    )
    
    if with_save_button and save_button_id:
        input_row = html.Div([
            dropdown,
            html.Button('+', id=save_button_id, n_clicks=0, style={'marginLeft': '10px'})
        ], style={
            'display': 'flex',
            'alignItems': 'center'
        })
    else:
        input_row = dropdown
    
    return html.Div([
        html.Label(label, style={'paddingLeft': '10px'}, title=tooltip),
        input_row
    ])


def create_slider_input(
    id: str,
    label: str,
    default: float,
    min_val: float,
    max_val: float,
    marks: Dict,
    step: Optional[float] = None,
    tooltip: str = '',
    with_save_button: bool = False,
    save_button_id: Optional[str] = None,
    persistence: bool = True
) -> html.Div:
    """
    Crée un slider avec label et tooltip optionnel.
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    slider = dcc.Slider(
        id=id,
        min=min_val,
        max=max_val,
        value=default,
        step=step,
        marks=marks,
        tooltip={"placement": "bottom", "always_visible": True},
        className='custom-slider',
        **persistence_props
    )
    
    components = [
        html.Label(label, style={'color': '#FF8C00', 'paddingBottom': '10px', 'paddingLeft': '10px'}),
        slider
    ]
    
    if with_save_button and save_button_id:
        components.append(html.Button('+', id=save_button_id, n_clicks=0, style={'marginTop': '6px'}))
    
    return html.Div(components, title=tooltip)


def create_saved_dropdown(
    id: str,
    placeholder: str = 'Options disponibles (fichier)',
    persistence: bool = True
) -> dcc.Dropdown:
    """
    Crée un dropdown pour les valeurs sauvegardées.
    """
    persistence_props = {'persistence': True, 'persistence_type': 'session'} if persistence else {}
    
    return dcc.Dropdown(
        id=id,
        options=[],
        value=[],
        multi=True,
        placeholder=placeholder,
        style={'width': '100%', 'marginTop': '6px', 'color': '#FF8C00'},
        **persistence_props
    )

