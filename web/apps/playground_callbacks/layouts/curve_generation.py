# -*- coding: utf-8 -*-
"""
Panel de génération de courbes synthétiques.
"""

from dash import dcc, html
import pandas as pd


def create_curve_generation_panel(start_date=None, end_date=None):
    """
    Crée le panneau de génération de courbe synthétique.
    
    Args:
        start_date: Date de début (défaut: aujourd'hui - 20 jours)
        end_date: Date de fin (défaut: aujourd'hui)
    
    Returns:
        html.Div contenant le panneau
    """
    if start_date is None or end_date is None:
        today = pd.Timestamp.today().normalize()
        start_date = (today - pd.Timedelta(days=20)).date()
        end_date = today.date()
    
    # Tooltips
    t_curve = 'Choisir la forme de la série synthétique (tendance, saisonnalité, etc.)'
    t_period = 'Période de génération des données'
    t_open = "Heure d'ouverture du marché (HH:MM)"
    t_close = 'Heure de fermeture du marché (HH:MM)'
    t_price = 'Prix initial de la série'
    t_trend = 'Force de la tendance directionnelle (pente)'
    t_seas = 'Amplitude de la saisonnalité intra‑journalière'
    t_sine = 'Période (en minutes) de la composante sinusoïdale'
    t_lunch = "Intensité de l'effet de pause déjeuner (réduction de volatilité)"
    t_seed = 'Seed aléatoire pour la reproductibilité (laisser vide pour aléatoire)'
    
    return html.Div([
        # Store pour mémoriser les ratios "souhaités" (utile quand #patterns/#amorces varie)
        dcc.Store(
            id='play_pattern_ratio_pref',
            storage_type='session',
            data={
                'patterns_par_amorce': 2,
                'amorces_par_pattern': 2,
            },
        ),
        html.H4('Génération de courbe', style={'color': '#FF8C00', 'marginBottom': '8px'}),
        
        # Type de courbe
        html.Div([
            html.Label('Type de courbe', title=t_curve),
            html.Div([
                dcc.Dropdown(
                    id='play_curve_type',
                    options=[
                        {'label': '🎲 Random walk', 'value': 'random_walk'},
                        {'label': '📈 Trend', 'value': 'trend'},
                        {'label': '🌊 Seasonal', 'value': 'seasonal'},
                        {'label': '🍽️ Lunch effect', 'value': 'lunch_effect'},
                        {'label': '〰️ Sinusoïdale', 'value': 'sinusoidale'},
                        {'label': '📊 Plateau (N niveaux)', 'value': 'plateau'},
                        {'label': '🧩 Pattern (amorce+motif)', 'value': 'pattern'},
                    ],
                    value='random_walk',
                    persistence=True, persistence_type='session',
                    style={'width': '100%', 'color': '#FF8C00'}
                )
            ], title=t_curve)
        ]),
        
        # Période
        html.Div([
            html.Label('Période', title=t_period),
            html.Div([
                dcc.DatePickerRange(
                    id='play_date_range',
                    start_date=start_date,
                    end_date=end_date,
                    display_format='YYYY-MM-DD'
                )
            ], title=t_period)
        ], style={'marginTop': '8px'}),
        
        # Heures ouverture/fermeture
        html.Div([
            html.Div([
                html.Label('Heure ouverture', title=t_open),
                html.Div(
                    dcc.Input(
                        id='play_open_time', value='09:30', type='text',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_open
                ),
            ]),
            html.Div([
                html.Label('Heure fermeture', title=t_close),
                html.Div(
                    dcc.Input(
                        id='play_close_time', value='16:00', type='text',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_close
                ),
            ]),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(140px, 1fr))', 'gap': '8px', 'marginTop': '8px'}),
        
        # Paramètres de courbe
        html.Div([
            html.Div([
                html.Label('Prix initial', title=t_price),
                html.Div(
                    dcc.Input(
                        id='play_base_price', value=100.0, type='number', step=1,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_price
                ),
            ]),
            html.Div([
                html.Label('Bruit', title='Bruit multiplicatif (0 = courbe parfaite, 0.001 = léger bruit)'),
                html.Div(
                    dcc.Input(
                        id='play_noise', value=0.0, type='number', step=0.0001, min=0,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title='Bruit multiplicatif'
                ),
            ]),
            html.Div([
                html.Label('Trend', title=t_trend),
                html.Div(
                    dcc.Input(
                        id='play_trend_strength', value=0.0, type='number', step=0.0001,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_trend
                ),
            ]),
            html.Div([
                html.Label('Amplitude', title=t_seas),
                html.Div(
                    dcc.Input(
                        id='play_seasonality_amp', value=0.20, type='number', step=0.01, min=0,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_seas
                ),
            ]),
            html.Div([
                html.Label('Période sinus', title=t_sine),
                html.Div(
                    dcc.Input(
                        id='play_sine_period', value=360, type='number', step=1, min=1,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_sine
                ),
            ], id='container_sine_period'),
            html.Div([
                html.Label('Reset daily', title='Si coché, la sinusoïde redémarre à 0 chaque matin'),
                dcc.Checklist(
                    id='play_sine_reset_daily',
                    options=[
                        {'label': ' Oui', 'value': True},
                    ],
                    value=[],
                    persistence=True,
                    persistence_type='session',
                    style={'color': '#FFF'},
                ),
            ], id='container_sine_reset'),
            html.Div([
                html.Label('Décalage phase (°/jour)', title='Décale la phase de la sinusoïde chaque jour (cumulatif)'),
                html.Div(
                    dcc.Input(
                        id='play_sine_phase_shift',
                        value=0.0,
                        type='number',
                        step=1,
                        style={'width': '100%'},
                        persistence=True,
                        persistence_type='session',
                    )
                ),
            ], id='container_sine_offset'),
            html.Div([
                html.Label('Nb plateaux', title='Nombre de plateaux pour la courbe Plateau'),
                html.Div(
                    dcc.Input(
                        id='play_nb_plateaux', value=3, type='number', step=1, min=2, max=10,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title='Nombre de niveaux de prix'
                ),
            ], id='container_nb_plateaux'),
            html.Div([
                html.Label('Lunch effect', title=t_lunch),
                html.Div(
                    dcc.Input(
                        id='play_lunch_strength', value=0.0, type='number', step=0.001, min=0,
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_lunch
                ),
            ], id='container_lunch_strength'),
            html.Div([
                html.Label('Seed', title=t_seed),
                html.Div(
                    dcc.Input(
                        id='play_seed', value=None, type='number',
                        style={'width': '100%'},
                        persistence=True, persistence_type='session'
                    ),
                    title=t_seed
                ),
            ]),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px', 'marginTop': '8px'}),
        
        # Section paramètres Pattern (visible uniquement pour type 'pattern')
        html.Div(id='play_pattern_params_container', children=[
            html.Hr(style={'borderColor': '#444', 'margin': '12px 0'}),
            html.Label('Paramètres Pattern', style={'color': '#FF8C00', 'fontWeight': 'bold', 'fontSize': '13px'}),
            html.Div([
                html.Div([
                    html.Label('Nb patterns', title='Nombre total de patterns différents dans le stock'),
                    html.Div(
                        dcc.Input(
                            id='play_nb_patterns', value=4, type='number', step=1, min=1, max=20,
                            style={'width': '100%'},
                            persistence=True, persistence_type='session'
                        ),
                        title='Nombre de motifs patterns disponibles'
                    ),
                ]),
                html.Div([
                    html.Label('Nb amorces', title='Nombre total d\'amorces différentes dans le stock'),
                    html.Div(
                        dcc.Input(
                            id='play_nb_amorces', value=4, type='number', step=1, min=1, max=20,
                            style={'width': '100%'},
                            persistence=True, persistence_type='session'
                        ),
                        title='Nombre d\'amorces disponibles'
                    ),
                ]),
                html.Div([
                    html.Label('Durée amorce (min)', title='Durée de la phase d\'amorce en début de journée (en minutes)'),
                    html.Div(
                        dcc.Input(
                            id='play_duree_amorce', value=60, type='number', step=5, min=5, max=180,
                            style={'width': '100%'},
                            persistence=True, persistence_type='session'
                        ),
                        title='Minutes de la phase d\'amorce'
                    ),
                ]),
                html.Div([
                    html.Label('Patterns/amorce', title='Nombre de patterns différents associés à chaque amorce'),
                    html.Div(
                        dcc.Input(
                            id='play_patterns_par_amorce', value=2, type='number', step=1, min=1, max=10,
                            style={'width': '100%'},
                            persistence=True, persistence_type='session'
                        ),
                        title='Chaque amorce peut mener à N patterns différents'
                    ),
                ]),
                html.Div([
                    html.Label('Amorces/pattern', title='Nombre d\'amorces différentes associées à chaque pattern'),
                    html.Div(
                        dcc.Input(
                            id='play_amorces_par_pattern', value=2, type='number', step=1, min=1, max=10,
                            style={'width': '100%'},
                            persistence=True, persistence_type='session'
                        ),
                        title='Chaque pattern peut suivre N amorces différentes'
                    ),
                ]),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(110px, 1fr))', 'gap': '8px', 'marginTop': '6px'}),
            html.Div(
                '💡 La même amorce peut conduire à patterns_par_amorce patterns différents. '
                'Un même pattern peut suivre amorces_par_pattern amorces différentes.',
                style={'fontSize': '11px', 'color': '#888', 'marginTop': '6px', 'fontStyle': 'italic'}
            ),
        ], style={'display': 'none'}),  # Caché par défaut, affiché via callback
        
        # Message d'aide dynamique
        html.Div(id='curve_info_msg', style={
            'marginTop': '8px',
            'padding': '8px',
            'backgroundColor': '#1a1a1a',
            'borderRadius': '4px',
            'fontSize': '12px'
        }),
        
        # Bouton générer
        html.Button(
            'Générer la courbe',
            id='play_generate',
            n_clicks=0,
            title='Générer une nouvelle série synthétique',
            style={
                'width': '100%',
                'marginTop': '8px',
                'backgroundColor': '#FF8C00',
                'color': 'white',
                'border': 'none',
                'borderRadius': '10px',
                'fontWeight': '600'
            }
        ),
        
        # Mini preview
        dcc.Graph(
            id='play_mini_preview',
            config={'displayModeBar': False},
            style={'height': '150px', 'marginTop': '10px'},
            figure={
                'data': [],
                'layout': {
                    'height': 1,
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)'
                }
            }
        )
    ], style={'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px'})

