from dash import dcc, html, Input, Output, State
import dash
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

# Forcer l'exécution CPU (évite les erreurs de contexte CUDA/XLA si les drivers GPU sont absents ou instables)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf
import time
from io import StringIO
import logging
import tempfile

from app import app, shM
from web.services.synthetic import generate_synthetic_timeseries, estimate_nb_quotes_per_day
from web.components.navigation import create_navigation, create_page_help

# Configuration du logging pour afficher les messages en console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from web.services.model_strategy import backtest_model_intraday
from web.services.sim_builders import (
    build_equity_figure,
    build_trades_table,
    build_daily_outputs,
    build_summary,
)

# Import des modèles Transformer/Hybride
try:
    from Models.transformer import (
        create_transformer_model,
        create_hybrid_lstm_transformer_model,
        get_model_architecture_info,
        get_custom_objects
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logging.warning("[Playground] Module transformer non disponible")

# Import de la configuration centralisée des modèles
from web.apps.model_config import (
    # Constantes par défaut
    DEFAULT_EPOCHS,
    DEFAULT_LOOK_BACK,
    DEFAULT_STRIDE,
    DEFAULT_NB_Y,
    DEFAULT_FIRST_MINUTES,
    DEFAULT_LSTM_UNITS as DEFAULT_UNITS,
    DEFAULT_LSTM_LAYERS as DEFAULT_LAYERS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_INITIAL_CASH,
    DEFAULT_TRADE_AMOUNT,
    DEFAULT_K_TRADES,
    TRAINING_GRAPH_UPDATE_INTERVAL_SECONDS,
    # Constantes Transformer
    DEFAULT_EMBED_DIM,
    DEFAULT_NUM_HEADS,
    DEFAULT_TRANSFORMER_LAYERS,
    DEFAULT_FF_MULTIPLIER,
    DEFAULT_DROPOUT,
    # Constantes Hybride
    DEFAULT_FUSION_MODE,
    DEFAULT_SPREAD_PCT,
    # Fonctions et définitions UI
    get_model_type_options,
    get_fusion_mode_options,
    MODEL_TYPES,
    TOOLTIPS,
)


# Dernier modèle entraîné dans le Playground (pour tests de généralisation)
play_last_model = None
play_last_model_meta = {}
play_last_model_path = None


def _get_symbols_options():
    try:
        df = shM.getAllShares()
        symbols = list(df['symbol'].values) if not df.empty else []
        return [{ 'label': s, 'value': s } for s in symbols]
    except Exception:
        return []


def _default_dates():
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=20)
    return start, today


def layout_content():
    start, end = _default_dates()
    
    # Définition des infobulles (titles) pour réutilisation (Label + Input)
    t_curve = 'Choisir la forme de la série synthétique (tendance, saisonnalité, etc.)'
    t_period = 'Période de génération des données'
    t_open = "Heure d'ouverture du marché (HH:MM)"
    t_close = 'Heure de fermeture du marché (HH:MM)'
    t_price = 'Prix initial de la série'
    t_vol = 'Amplitude aléatoire minute à minute (volatilité)'
    t_trend = 'Force de la tendance directionnelle (pente)'
    t_seas = 'Amplitude de la saisonnalité intra‑journalière'
    t_sine = 'Période (en minutes) de la composante sinusoïdale'
    t_lunch = 'Intensité de l’effet de pause déjeuner (réduction de volatilité)'
    t_noise = 'Bruit additif supplémentaire (aléatoire)'
    t_seed = 'Seed aléatoire pour la reproductibilité (laisser vide pour aléatoire)'
    
    t_lookback = 'Taille de la fenêtre d’entrée (en points/minutes)'
    t_stride = "Pas d'échantillonnage pour la fenêtre d'entrée (ex: 5 = 1 point toutes les 5 min)"
    t_nby = 'Nombre de points futurs à prédire (répartis uniformément sur le reste de la journée)'
    t_predtype = "Type de cible à prédire : Variation (Return) ou Prix Normalisé (Price)"
    t_da = 'Activer la métrique Directional Accuracy (pourcentage de bonnes directions)'
    t_loss_type = '''Type de fonction de perte (Loss) pour l'entraînement:
• MSE (défaut): Mean Squared Error - erreur quadratique moyenne. Simple mais donne des valeurs très petites si les variations sont faibles.
• Scaled MSE (×100): MSE multiplié par 100 - les targets sont multipliées par 100, le loss est plus lisible (~0.01-1.0 au lieu de ~0.0001).
• MAE: Mean Absolute Error - erreur absolue moyenne. Plus robuste aux outliers, plus facile à interpréter (même unité que les targets).'''
    t_first = "Nombre de minutes d'observation en début de journée (Input du modèle)"
    t_units = 'Nombre de neurones par couche LSTM'
    t_layers = 'Nombre de couches LSTM empilées'
    t_lr = "Vitesse d'apprentissage (Learning Rate)"
    t_epochs = "Nombre d'itérations complètes sur le jeu d'entraînement"
    
    t_symbol = 'Filtrer les modèles sauvegardés par symbole'
    t_saved = 'Sélectionner un modèle déjà entraîné'
    
    t_cash = 'Capital de départ pour la simulation'
    t_trade_amt = 'Montant engagé par trade'
    t_ktrades = 'Nombre maximum de trades simultanés/journaliers'
    t_spread = 'Spread bid-ask en % appliqué à chaque trade (coût de transaction)'
    
    # Tooltips Transformer (les tooltips sont maintenant dans TOOLTIPS de model_config.py)
    # Variables locales conservées pour compatibilité avec le code existant
    t_embed_dim = TOOLTIPS['embed_dim']
    t_num_heads = TOOLTIPS['num_heads']
    t_trans_layers = TOOLTIPS['transformer_layers']
    t_ff_mult = TOOLTIPS['ff_multiplier']
    t_dropout = TOOLTIPS['dropout']

    help_text = """
### Playground (Bac à Sable)

Cet outil est un laboratoire expérimental pour comprendre et tester le fonctionnement de l'IA sur des données de marché synthétiques.

---

#### 1. Génération de Courbe

Créez des séries temporelles artificielles pour voir si le modèle est capable d'apprendre des motifs simples.

**Types de courbes disponibles :**
*   **Random walk** : Marche aléatoire pure (imprévisible par nature)
*   **Trend** : Tendance directionnelle progressive (haussière ou baissière)
*   **Seasonal** : Cycle sinusoïdal intra-journalier
*   **Lunch effect** : Baisse de volatilité entre 12h et 14h
*   **Sinusoïdale** : Oscillation périodique régulière
*   **📊 Plateau (3 niveaux)** : 3 paliers fixes qui se répètent chaque jour :
    - **Matin** (1er tiers) : Prix de base
    - **Midi** (2ème tiers) : Prix + amplitude
    - **Après-midi** (3ème tiers) : Prix - amplitude/2
    - *Idéal pour tester si le modèle détecte les patterns répétitifs !*

---

#### 2. Les 3 Types de Modèles IA

---

##### 🔄 LSTM (Long Short-Term Memory)

**Qu'est-ce que c'est ?**
Un réseau de neurones récurrent spécialement conçu pour les séquences temporelles. Le "L" de Long signifie qu'il peut retenir des informations sur de longues périodes.

**Comment ça fonctionne ?**
1. Le LSTM lit la séquence **point par point**, de gauche à droite
2. À chaque étape, il décide :
   - 🚪 **Forget Gate** : Quelles informations passées oublier ?
   - 📥 **Input Gate** : Quelles nouvelles informations mémoriser ?
   - 📤 **Output Gate** : Que retourner comme résultat ?
3. Il maintient une **mémoire interne** (cell state) qui traverse toute la séquence

**Forces :**
- ✅ Excellent pour les **dépendances séquentielles locales** (le prix d'il y a 5 min influence celui de maintenant)
- ✅ Moins gourmand en mémoire que le Transformer
- ✅ Bien adapté aux séries temporelles régulières

**Faiblesses :**
- ❌ Traitement **séquentiel** (lent à entraîner)
- ❌ Difficultés avec les **très longues séquences** (> 200 points)
- ❌ Ne voit pas les relations entre points éloignés facilement

**Paramètres clés :**
- `Unités LSTM` : Plus il y en a, plus le modèle peut mémoriser (mais risque de sur-apprentissage)
- `Couches` : Empiler plusieurs LSTM permet d'abstraire à différents niveaux

---

##### 🎯 Transformer (Attention Multi-Têtes)

**Qu'est-ce que c'est ?**
L'architecture révolutionnaire derrière ChatGPT, BERT, etc. Utilise le mécanisme d'**attention** pour comprendre les relations entre tous les points de la séquence simultanément.

**Comment ça fonctionne ?**
1. **Encodage positionnel** : Ajoute l'information "où" chaque point se situe dans la séquence
2. **Self-Attention** : Pour chaque point, calcule son "attention" vers tous les autres :
   - "Le prix à 10h30 est-il corrélé au prix à 9h45 ?"
   - "L'ouverture prédit-elle la fermeture ?"
3. **Multi-Head** : Plusieurs "têtes" regardent différents aspects en parallèle :
   - Tête 1 : tendance générale
   - Tête 2 : volatilité récente
   - Tête 3 : patterns cycliques
   - etc.
4. **Feed-Forward** : Réseau dense pour combiner les informations

**Forces :**
- ✅ Voit **toutes les relations** dans la séquence d'un coup
- ✅ Traitement **parallèle** (rapide sur GPU)
- ✅ Excellent pour les **patterns complexes et globaux**
- ✅ Scalable (fonctionne bien avec beaucoup de données)

**Faiblesses :**
- ❌ Gourmand en **mémoire** (O(n²) avec la longueur)
- ❌ Nécessite plus de **données** pour bien apprendre
- ❌ Peut "sur-interpréter" du bruit comme des patterns

**Paramètres clés :**
- `Embed dim` : Taille des vecteurs internes (64-256 typique)
- `Num heads` : Nombre de perspectives d'attention parallèles
- `Layers` : Profondeur du réseau (plus = plus abstrait)
- `FF multiplier` : Taille de la couche Feed-Forward (généralement 4×embed_dim)

---

##### 🔀 Hybride LSTM + Transformer

**Qu'est-ce que c'est ?**
Le meilleur des deux mondes ! Combine la mémoire séquentielle du LSTM avec la vision globale du Transformer.

**Comment ça fonctionne ?**
1. **Branche LSTM** : Traite la séquence point par point
   - Capture : tendance récente, momentum, patterns locaux
   - Produit un vecteur "résumé séquentiel"

2. **Branche Transformer** : Traite toute la séquence en parallèle
   - Capture : corrélations à distance, patterns cycliques, anomalies
   - Produit un vecteur "résumé global"

3. **Fusion** : Combine les deux représentations
   - **Concat** : Met les deux vecteurs bout à bout [LSTM | Transformer]
   - **Add** : Additionne les représentations (après projection)
   - **Attention** : Le LSTM "interroge" le Transformer via cross-attention

4. **Couches de sortie** : Génère les prédictions finales

**Quand l'utiliser ?**
- Quand les données ont à la fois :
  - Des **patterns locaux** (momentum court terme)
  - Des **patterns globaux** (saisonnalité, corrélations long terme)
- Quand un modèle seul ne suffit pas

**Modes de fusion :**
- **Concat** : Simple et robuste, double la dimension
- **Add** : Plus compact, force les représentations à être compatibles
- **Attention** : Le plus expressif, le LSTM peut "choisir" quoi prendre du Transformer

---

#### 3. Paramètres de Données

- `look_back` : Combien de minutes passées le modèle voit (60 = 1h)
- `stride` : Échantillonnage (stride=5 → 1 point toutes les 5 min)
- `nb_y` : Combien de points futurs prédire
- `Premières minutes` : Période d'observation avant de trader

---

#### 💡 Conseils pour la courbe Plateau

Pour tester efficacement avec la courbe **Plateau** :

**Paramètres de courbe recommandés :**
- Bruit : **0** (courbe parfaite, déterministe)
- Amplitude : **0.20** (20% entre niveaux = facile à apprendre)
- Nb plateaux : **3** (ou plus pour augmenter la difficulté)
- Tous les autres à 0

**Paramètres modèle recommandés :**
- Type : **LSTM** (suffisant pour ce pattern simple)
- Unités : **32** (64 est trop, surapprentissage)
- Couches : **1** (2 couches = trop complexe)
- Learning rate : **0.01** (plus agressif pour converger vite)
- Epochs : **50-100** (suffisant)
- Type prédiction : **Prix** (plus stable que Retours)

**Objectif de loss :**
- Avec 20% d'amplitude, une loss < **0.001** = très bon
- Une loss de **0.0001** = quasi-parfait

---

#### 4. Types de Loss (Fonction de Perte)

Le choix de la loss influence l'entraînement et la lisibilité des métriques :

| Type | Description | Avantages | Inconvénients |
|------|-------------|-----------|---------------|
| **MSE** | Mean Squared Error (défaut) | Standard, stable | Valeurs très petites (10⁻⁶) si variations faibles |
| **Scaled MSE** | MSE × 100 | Loss lisible (~0.01-1.0), même comportement que MSE | Prédictions à rescaler mentalement |
| **MAE** | Mean Absolute Error | Robuste aux outliers, même unité que les targets | Moins pénalisant pour les grosses erreurs |

**Recommandation :** Utilisez **Scaled MSE** si le loss standard est illisible (< 0.0001).

---

#### 5. Stratégies de Trading (Backtest)

| Stratégie | Description | Quand l'utiliser |
|-----------|-------------|------------------|
| **📈 LONG** | Acheter si hausse prédite → Vendre plus tard | Marché haussier ou pattern de hausse |
| **📉 SHORT** | Vendre si baisse prédite → Racheter moins cher | Marché baissier ou pattern de baisse |
| **📊 LONG & SHORT** | Les deux selon la prédiction | Maximum d'opportunités |

- Les trades ne se chevauchent **jamais** sur une même journée
- Chaque jour, jusqu'à **K trades** sont exécutés parmi les prédictions les plus fortes
- Le **spread** est appliqué sur chaque trade (coût réaliste)

---

#### 6. Résultats

- **Série synthétique** : La courbe générée avec les prédictions
- **Équité** : Évolution du portefeuille selon la stratégie
- **Tableau des trades** : Détail avec direction (📈/📉), heures entrée/sortie, P&L
- **Historique** : Loss (échelle log) et Directional Accuracy pendant l'entraînement
"""

    return html.Div([
        create_page_help("Aide Playground", help_text),
        html.H3('Playground', style={ 'color': '#FF8C00' }),

        dcc.Store(id='play_df_store', storage_type='session'),

        html.Div([
            html.Div([
                html.H4('Génération de courbe', style={ 'color': '#FF8C00', 'marginBottom': '8px' }),
                html.Div([
                    html.Label('Type de courbe', title=t_curve),
                    html.Div([
                        dcc.Dropdown(
                            id='play_curve_type',
                            options=[
                                { 'label': '🎲 Random walk', 'value': 'random_walk' },
                                { 'label': '📈 Trend', 'value': 'trend' },
                                { 'label': '🌊 Seasonal', 'value': 'seasonal' },
                                { 'label': '🍽️ Lunch effect', 'value': 'lunch_effect' },
                                { 'label': '〰️ Sinusoïdale', 'value': 'sinusoidale' },
                                { 'label': '📊 Plateau (N niveaux)', 'value': 'plateau' },
                            ],
                            value='random_walk',
                            persistence=True, persistence_type='session',
                            style={ 'width': '100%', 'color': '#FF8C00' }
                        )
                    ], title=t_curve)
                ]),
                html.Div([
                    html.Label('Période', title=t_period),
                    html.Div([
                        dcc.DatePickerRange(
                            id='play_date_range',
                            start_date=start.date(),
                            end_date=end.date(),
                            display_format='YYYY-MM-DD'
                        )
                    ], title=t_period)
                ], style={'marginTop': '8px'}),
                
                html.Div([
                    html.Div([
                        html.Label('Heure ouverture', title=t_open),
                        html.Div(dcc.Input(id='play_open_time', value='09:30', type='text', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_open),
                    ]),
                    html.Div([
                        html.Label('Heure fermeture', title=t_close),
                        html.Div(dcc.Input(id='play_close_time', value='16:00', type='text', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_close),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(2, minmax(140px, 1fr))', 'gap': '8px', 'marginTop': '8px' }),
                
                html.Div([
                    html.Div([
                        html.Label('Prix initial', title=t_price),
                        html.Div(dcc.Input(id='play_base_price', value=100.0, type='number', step=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_price),
                    ]),
                    html.Div([
                        html.Label('Bruit', title='Bruit multiplicatif (0 = courbe parfaite, 0.001 = léger bruit)'),
                        html.Div(dcc.Input(id='play_noise', value=0.0, type='number', step=0.0001, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title='Bruit multiplicatif'),
                    ]),
                    html.Div([
                        html.Label('Trend', title=t_trend),
                        html.Div(dcc.Input(id='play_trend_strength', value=0.0, type='number', step=0.0001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trend),
                    ]),
                    html.Div([
                        html.Label('Amplitude', title=t_seas),
                        html.Div(dcc.Input(id='play_seasonality_amp', value=0.20, type='number', step=0.01, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_seas),
                    ]),
                    html.Div([
                        html.Label('Période sinus', title=t_sine),
                        html.Div(dcc.Input(id='play_sine_period', value=360, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_sine),
                    ]),
                    html.Div([
                        html.Label('Nb plateaux', title='Nombre de plateaux pour la courbe Plateau'),
                        html.Div(dcc.Input(id='play_nb_plateaux', value=3, type='number', step=1, min=2, max=10, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title='Nombre de niveaux de prix'),
                    ]),
                    html.Div([
                        html.Label('Lunch effect', title=t_lunch),
                        html.Div(dcc.Input(id='play_lunch_strength', value=0.0, type='number', step=0.001, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lunch),
                    ]),
                    html.Div([
                        html.Label('Seed', title=t_seed),
                        html.Div(dcc.Input(id='play_seed', value=None, type='number', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_seed),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px', 'marginTop': '8px' }),
                
                # Message d'aide selon le type de courbe
                html.Div(id='curve_info_msg', style={ 'marginTop': '8px', 'padding': '8px', 'backgroundColor': '#1a1a1a', 'borderRadius': '4px', 'fontSize': '12px' }),
                
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
                # Petit graphe de preview
                dcc.Graph(id='play_mini_preview', config={'displayModeBar': False}, style={'height': '150px', 'marginTop': '10px'}, figure={'data': [], 'layout': {'height': 1, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}})
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),

            html.Div([
                html.H4('Modèle et backtest', style={ 'color': '#FF8C00', 'marginBottom': '8px' }),
                dcc.RadioItems(
                    id='play_model_mode',
                    options=[
                        { 'label': 'Nouveau modèle', 'value': 'new' },
                        { 'label': 'Modèle sauvegardé (BDD)', 'value': 'saved' },
                    ],
                    value='new',
                    labelStyle={ 'display': 'inline-block', 'marginRight': '12px' },
                ),

                # Sélecteur de type de modèle (visible seulement en mode "new")
                html.Div([
                    html.Label('Type de modèle IA', title=TOOLTIPS['model_type'], style={ 'fontWeight': 'bold', 'marginTop': '8px' }),
                    dcc.Dropdown(
                        id='play_model_type',
                        options=get_model_type_options(include_gru=False, include_hybrid=True),
                        value='lstm',
                        persistence=True, persistence_type='session',
                        style={ 'width': '100%', 'color': '#FF8C00' }
                    ),
                ], id='panel_model_type_selector', style={ 'marginBottom': '12px' }),

                # Paramètres de données (communs à tous les modèles)
                html.Div([
                    html.Label('📊 Paramètres de données', style={ 'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px' }),
                ], style={ 'marginTop': '8px' }),

                html.Div([
                    html.Div([
                            html.Label('look_back (Window)', title=t_lookback),
                        html.Div(dcc.Input(id='play_look_back', value='60', type='text', style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lookback),
                    ]),
                    html.Div([
                        html.Label('stride', title=t_stride),
                        html.Div(dcc.Input(id='play_stride', value=1, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_stride),
                    ]),
                    html.Div([
                        html.Label('nb_y (horizon)', title=t_nby),
                        html.Div([
                            dcc.Slider(id='play_nb_y', min=1, max=60, step=1, value=5, marks={ 1: '1', 60: '60' }, persistence=True, persistence_type='session'),
                        ], title=t_nby),
                        html.Div(id='play_nb_y_value', style={ 'marginTop': '4px', 'color': '#FFFFFF', 'fontSize': '12px' }),
                    ]),
                    html.Div([
                        html.Label('Premières minutes (obs)', title=t_first),
                        html.Div(dcc.Input(id='play_first_minutes', value=60, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_first),
                    ]),
                    html.Div([
                        html.Label('Type de prédiction', title=t_predtype),
                        html.Div([
                            dcc.RadioItems(
                                id='play_prediction_type',
                                options=[
                                    { 'label': 'Variation (%)', 'value': 'return' },
                                    { 'label': 'Prix', 'value': 'price' },
                                    { 'label': 'Signal / Index', 'value': 'signal' },
                                ],
                                value='price',
                                labelStyle={ 'display': 'inline-block', 'marginRight': '8px' },
                                persistence=True, persistence_type='session',
                            ),
                        ], title=t_predtype),
                    ]),
                    html.Div([
                        html.Label('Directional Accuracy', title=t_da),
                        html.Div([
                            dcc.RadioItems(
                                id='play_use_directional_accuracy',
                                options=[
                                    { 'label': 'Oui', 'value': True },
                                    { 'label': 'Non', 'value': False },
                                ],
                                value=True,
                                labelStyle={ 'display': 'inline-block', 'marginRight': '8px' },
                                persistence=True, persistence_type='session',
                            ),
                        ], title=t_da),
                    ]),
                    html.Div([
                        html.Label('Type de Loss', title=t_loss_type),
                        dcc.Dropdown(
                            id='play_loss_type',
                            options=[
                                { 'label': 'MSE (défaut)', 'value': 'mse' },
                                { 'label': 'Scaled MSE (×100)', 'value': 'scaled_mse' },
                                { 'label': 'MAE', 'value': 'mae' },
                            ],
                            value='mse',
                            persistence=True, persistence_type='session',
                            style={ 'width': '100%', 'color': '#FF8C00' }
                        ),
                    ]),
                ], id='panel_play_data_params', style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(130px, 1fr))', 'gap': '8px' }),

                # ========== Paramètres LSTM ==========
                    html.Div([
                    html.Label('🔄 Architecture LSTM', style={ 'fontWeight': 'bold', 'color': '#1f77b4', 'marginBottom': '4px', 'marginTop': '12px' }),
                ], id='label_lstm_params'),
                html.Div([
                    html.Div([
                        html.Label('Unités LSTM', title=t_units),
                        html.Div(dcc.Input(id='play_units', value=64, type='number', step=1, min=4, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_units),
                    ]),
                    html.Div([
                        html.Label('Couches LSTM', title=t_layers),
                        html.Div(dcc.Input(id='play_layers', value=1, type='number', step=1, min=1, max=4, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_layers),
                    ]),
                ], id='panel_lstm_params', style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px' }),

                # ========== Paramètres Transformer ==========
                html.Div([
                    html.Label('🎯 Architecture Transformer', style={ 'fontWeight': 'bold', 'color': '#2ca02c', 'marginBottom': '4px', 'marginTop': '12px' }),
                ], id='label_transformer_params', style={ 'display': 'none' }),
                html.Div([
                    html.Div([
                        html.Label('Embed dim', title=t_embed_dim),
                        html.Div(dcc.Input(id='play_embed_dim', value=64, type='number', step=8, min=16, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_embed_dim),
                    ]),
                    html.Div([
                        html.Label('Num heads', title=t_num_heads),
                        html.Div(dcc.Input(id='play_num_heads', value=4, type='number', step=1, min=1, max=16, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_num_heads),
                    ]),
                    html.Div([
                        html.Label('Transformer layers', title=t_trans_layers),
                        html.Div(dcc.Input(id='play_transformer_layers', value=2, type='number', step=1, min=1, max=6, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trans_layers),
                    ]),
                    html.Div([
                        html.Label('FF multiplier', title=t_ff_mult),
                        html.Div(dcc.Input(id='play_ff_multiplier', value=4, type='number', step=1, min=1, max=8, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_ff_mult),
                    ]),
                    html.Div([
                        html.Label('Dropout', title=t_dropout),
                        html.Div(dcc.Input(id='play_dropout', value=0.1, type='number', step=0.05, min=0.0, max=0.5, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_dropout),
                    ]),
                ], id='panel_transformer_params', style={ 'display': 'none', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px' }),

                # ========== Paramètres Hybride (LSTM + Transformer) ==========
                html.Div([
                    html.Label('🔀 Architecture Hybride', style={ 'fontWeight': 'bold', 'color': '#9467bd', 'marginBottom': '4px', 'marginTop': '12px' }),
                ], id='label_hybrid_params', style={ 'display': 'none' }),
                html.Div([
                    html.Div([
                        html.Label('LSTM units', title=t_units),
                        html.Div(dcc.Input(id='play_hybrid_lstm_units', value=64, type='number', step=8, min=8, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_units),
                    ]),
                    html.Div([
                        html.Label('LSTM layers', title=t_layers),
                        html.Div(dcc.Input(id='play_hybrid_lstm_layers', value=1, type='number', step=1, min=1, max=3, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_layers),
                    ]),
                    html.Div([
                        html.Label('Embed dim', title=t_embed_dim),
                        html.Div(dcc.Input(id='play_hybrid_embed_dim', value=64, type='number', step=8, min=16, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_embed_dim),
                    ]),
                    html.Div([
                        html.Label('Trans. heads', title=t_num_heads),
                        html.Div(dcc.Input(id='play_hybrid_num_heads', value=4, type='number', step=1, min=1, max=8, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_num_heads),
                    ]),
                    html.Div([
                        html.Label('Trans. layers', title=t_trans_layers),
                        html.Div(dcc.Input(id='play_hybrid_trans_layers', value=1, type='number', step=1, min=1, max=4, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trans_layers),
                    ]),
                    html.Div([
                        html.Label('Fusion mode', title=TOOLTIPS['fusion_mode']),
                        dcc.Dropdown(
                            id='play_fusion_mode',
                            options=get_fusion_mode_options(),
                            value=DEFAULT_FUSION_MODE,
                            persistence=True, persistence_type='session',
                            style={ 'width': '100%', 'color': '#FF8C00' }
                        ),
                    ]),
                    html.Div([
                        html.Label('Dropout', title=t_dropout),
                        html.Div(dcc.Input(id='play_hybrid_dropout', value=0.1, type='number', step=0.05, min=0.0, max=0.5, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_dropout),
                    ]),
                ], id='panel_hybrid_params', style={ 'display': 'none', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px' }),

                # ========== Paramètres d'entraînement (communs) ==========
                html.Div([
                    html.Label('⚙️ Entraînement', style={ 'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px', 'marginTop': '12px' }),
                ]),
                html.Div([
                    html.Div([
                        html.Label('Learning rate', title=t_lr),
                        html.Div(dcc.Input(id='play_lr', value=0.001, type='number', step=0.0001, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_lr),
                    ]),
                    html.Div([
                        html.Label('Epochs', title=t_epochs),
                        html.Div(dcc.Input(id='play_epochs', value=5, type='number', step=1, min=1, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_epochs),
                    ]),
                ], id='panel_play_new', style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px' }),

                # ========== Section Modèles sauvegardés (BDD) ==========
                html.Div([
                    html.Label('📦 Charger depuis la BDD', style={ 'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '8px' }),
                    html.Div([
                        html.Div([
                            html.Label('Filtrer par type', title='Filtrer les modèles par architecture'),
                            dcc.Dropdown(
                                id='play_saved_model_type_filter',
                                options=[
                                    { 'label': 'Tous les types', 'value': 'all' },
                                    { 'label': f"{MODEL_TYPES['lstm']['icon']} {MODEL_TYPES['lstm']['short_label']}", 'value': 'lstm' },
                                    { 'label': f"{MODEL_TYPES['transformer']['icon']} {MODEL_TYPES['transformer']['short_label']}", 'value': 'transformer' },
                                    { 'label': f"{MODEL_TYPES['hybrid']['icon']} {MODEL_TYPES['hybrid']['short_label']}", 'value': 'hybrid' },
                                ],
                                value='all',
                                persistence=True, persistence_type='session',
                                style={ 'width': '100%', 'color': '#FF8C00' }
                            ),
                        ]),
                        html.Div([
                            html.Label('Symbole (optionnel)', title=t_symbol),
                            html.Div(dcc.Dropdown(id='play_symbol', options=_get_symbols_options(), placeholder='Tous les symboles', style={ 'width': '100%', 'color': '#FF8C00' }, persistence=True, persistence_type='session'), title=t_symbol),
                        ]),
                    ], style={ 'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '8px' }),
                    html.Div([
                        html.Label('Modèle sauvegardé', title=t_saved),
                        html.Div(dcc.Dropdown(id='play_saved_model', options=[], placeholder='Choisir un modèle', style={ 'width': '100%', 'color': '#FF8C00' }, persistence=True, persistence_type='session'), title=t_saved),
                    ], style={ 'marginTop': '8px' }),
                    html.Div(id='play_saved_model_info', style={ 'marginTop': '8px', 'color': '#888', 'fontSize': '12px' }),
                ], id='panel_play_saved', style={ 'display': 'none' }),

                # ========== Bouton Entraîner (avant Simulation Financière) ==========
                html.Div([
                    html.Button('🎯 Entraîner le modèle', id='play_train_backtest', n_clicks=0, style={ 'width': '100%', 'backgroundColor': '#4CAF50', 'padding': '12px', 'fontSize': '14px', 'fontWeight': 'bold' }),
                ], id='panel_play_btn_train', style={ 'marginTop': '12px', 'marginBottom': '8px' }),

                html.Hr(),
                
                # ========== Simulation Financière ==========
                html.Div([
                    html.Label('💰 Simulation Financière (Backtest)', style={ 'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px' }),
                ]),
                html.Div([
                    html.Div([
                        html.Label('Capital initial (€)', title=t_cash),
                        html.Div(dcc.Input(id='play_initial_cash', value=10_000, type='number', step=100, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_cash),
                    ]),
                    html.Div([
                        html.Label('Montant par trade (€)', title=t_trade_amt),
                        html.Div(dcc.Input(id='play_trade_amount', value=1_000, type='number', step=50, min=0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_trade_amt),
                    ]),
                    html.Div([
                        html.Label('K trades/jour', title=t_ktrades),
                        html.Div(dcc.Input(id='play_k_trades', value=2, type='number', step=1, min=1, max=10, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_ktrades),
                    ]),
                    html.Div([
                        html.Label('Spread (%)', title=t_spread),
                        html.Div(dcc.Input(id='play_spread_pct', value=0.0, type='number', step=0.01, min=0.0, max=1.0, style={ 'width': '100%' }, persistence=True, persistence_type='session'), title=t_spread),
                    ]),
                    html.Div([
                        html.Label('Stratégie', title='LONG = acheter puis vendre (gagner si hausse). SHORT = vendre puis racheter (gagner si baisse). LONG&SHORT = les deux selon la prédiction.'),
                        dcc.Dropdown(
                            id='play_strategy',
                            options=[
                                { 'label': '📈 LONG (hausse)', 'value': 'long' },
                                { 'label': '📉 SHORT (baisse)', 'value': 'short' },
                                { 'label': '📊 LONG & SHORT', 'value': 'both' },
                            ],
                            value='long',
                            persistence=True, persistence_type='session',
                            style={ 'width': '100%', 'color': '#FF8C00' }
                        ),
                    ]),
                ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))', 'gap': '8px' }),

                html.Div([
                    html.Button(
                        '📈 Lancer le Backtest',
                        id='play_run_backtest',
                        n_clicks=0,
                        style={
                            'width': '100%',
                            'backgroundColor': '#2196F3',
                            'color': '#000000',
                            'padding': '12px',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'border': '1px solid #FFFFFF',
                        },
                        disabled=True,
                    ),
                ], id='panel_play_btn_new', style={ 'marginTop': '12px' }),

                html.Div([
                    html.Button('Backtester modèle sauvegardé', id='play_backtest_saved', n_clicks=0, style={ 'width': '100%' }),
                ], id='panel_play_btn_saved', style={ 'display': 'none' }),
                
                # Stores pour les prédictions
                dcc.Store(id='play_predictions_store', storage_type='memory'),
                html.Hr(),
                html.Div([
                    html.H4('Suivi entraînement', style={ 'color': '#FF8C00' }),
                    html.Div(id='play_training_progress', style={ 'marginBottom': '8px' }),
                    dcc.Graph(
                        id='play_training_history', style={ 'height': '300px' }, config={ 'responsive': False },
                        figure={ 'data': [], 'layout': { 'template': 'plotly_dark', 'paper_bgcolor': '#000', 'plot_bgcolor': '#000', 'font': { 'color': '#FFF' }, 'title': 'En attente d\'entraînement...', 'height': 280 } }
                    ),
                ], style={ 'marginTop': '12px' }),
                html.Div([
                    html.H4('Test de généralisation', style={ 'color': '#FF8C00', 'marginTop': '12px' }),
                    html.Button(
                        '🧪 Tester la généralisation sur la courbe actuelle',
                        id='play_test_generalization',
                        n_clicks=0,
                        style={
                            'width': '100%',
                            'backgroundColor': '#FF8C00',
                            'color': '#000000',
                            'padding': '10px',
                            'fontSize': '14px',
                            'fontWeight': 'bold',
                            'border': 'none',
                            'borderRadius': '8px',
                            'marginTop': '8px'
                        }
                    ),
                    html.Div(
                        id='play_gen_summary',
                        style={
                            'marginTop': '8px',
                            'color': '#CCCCCC',
                            'fontSize': '12px'
                        }
                    )
                ], style={ 'marginTop': '4px' }),
                # Stores pour modèle en mémoire
                dcc.Store(id='play_model_ready', storage_type='memory', data=False),
                dcc.Store(id='play_model_path', storage_type='memory'),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
        ], style={ 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '12px' }),

        html.Div([
            html.Div([
                html.H4('Série synthétique & Segments', style={ 'color': '#FF8C00' }),
                dcc.Graph(
                    id='play_segments_graph', style={ 'height': '450px' }, config={ 'responsive': False },
                    figure={ 'data': [], 'layout': { 'template': 'plotly_dark', 'paper_bgcolor': '#000', 'plot_bgcolor': '#000', 'font': { 'color': '#FFF' }, 'title': 'Cliquer sur "Générer la courbe"', 'height': 420 } }
                ),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
            html.Div([
                html.H4('Équité', style={ 'color': '#FF8C00' }),
                dcc.Graph(
                    id='play_equity_graph', style={ 'height': '420px' }, config={ 'responsive': False },
                    figure={ 'data': [], 'layout': { 'template': 'plotly_dark', 'paper_bgcolor': '#000', 'plot_bgcolor': '#000', 'font': { 'color': '#FFF' }, 'title': 'En attente de backtest...', 'height': 400 } }
                ),
                html.Div(id='play_trades_table', style={ 'marginTop': '8px' }),
                html.Div(id='play_summary', style={ 'marginTop': '8px' }),
            ], style={ 'backgroundColor': '#2E2E2E', 'padding': '12px', 'borderRadius': '8px' }),
        ], style={ 'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginTop': '12px' }),

        create_navigation()
    ], style={ 'backgroundColor': 'black', 'padding': '20px', 'minHeight': '100vh' })


layout = layout_content()


@app.callback(
    [
        Output('panel_play_new', 'style'),
        Output('panel_play_saved', 'style'),
        Output('panel_play_btn_new', 'style'),
        Output('panel_play_btn_saved', 'style'),
        Output('panel_model_type_selector', 'style'),
        Output('panel_play_data_params', 'style'),
        Output('label_lstm_params', 'style'),
        Output('panel_lstm_params', 'style'),
        Output('label_transformer_params', 'style'),
        Output('panel_transformer_params', 'style'),
        Output('label_hybrid_params', 'style'),
        Output('panel_hybrid_params', 'style'),
    ],
    [Input('play_model_mode', 'value')]
)
def toggle_play_panels(mode):
    show_grid = { 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px' }
    show_data_grid = { 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(130px, 1fr))', 'gap': '8px' }
    hide = { 'display': 'none' }
    show_btn = { 'display': 'block' }
    show_block = { 'display': 'block', 'marginBottom': '12px' }
    show_label = { 'display': 'block' }
    
    if mode == 'saved':
        # Mode sauvegardé: cacher tous les panels de paramètres, afficher panel sauvegardé
        return (
            hide,  # panel_play_new
            { 'display': 'block' },  # panel_play_saved
            hide,  # panel_play_btn_new
            show_btn,  # panel_play_btn_saved
            hide,  # panel_model_type_selector
            hide,  # panel_play_data_params
            hide,  # label_lstm_params
            hide,  # panel_lstm_params
            hide,  # label_transformer_params
            hide,  # panel_transformer_params
            hide,  # label_hybrid_params
            hide,  # panel_hybrid_params
        )
    
    # Mode nouveau modèle: afficher le sélecteur de type et les paramètres LSTM par défaut
    return (
        show_grid,  # panel_play_new
        hide,  # panel_play_saved
        show_btn,  # panel_play_btn_new
        hide,  # panel_play_btn_saved
        show_block,  # panel_model_type_selector
        show_data_grid,  # panel_play_data_params
        show_label,  # label_lstm_params (visible par défaut)
        show_grid,  # panel_lstm_params (visible par défaut)
        hide,  # label_transformer_params
        hide,  # panel_transformer_params
        hide,  # label_hybrid_params
        hide,  # panel_hybrid_params
    )


@app.callback(
    [
        Output('label_lstm_params', 'style', allow_duplicate=True),
        Output('panel_lstm_params', 'style', allow_duplicate=True),
        Output('label_transformer_params', 'style', allow_duplicate=True),
        Output('panel_transformer_params', 'style', allow_duplicate=True),
        Output('label_hybrid_params', 'style', allow_duplicate=True),
        Output('panel_hybrid_params', 'style', allow_duplicate=True),
    ],
    [Input('play_model_type', 'value')],
    prevent_initial_call=True
)
def toggle_model_type_params(model_type):
    """Affiche les paramètres correspondant au type de modèle sélectionné."""
    show_grid = { 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(100px, 1fr))', 'gap': '8px' }
    show_grid_lstm = { 'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px' }
    hide = { 'display': 'none' }
    show_label = { 'display': 'block' }
    
    if model_type == 'transformer':
        return hide, hide, show_label, show_grid, hide, hide
    elif model_type == 'hybrid':
        return hide, hide, hide, hide, show_label, show_grid
    else:  # lstm par défaut
        return show_label, show_grid_lstm, hide, hide, hide, hide


@app.callback(
    Output('curve_info_msg', 'children'),
    [Input('play_curve_type', 'value')]
)
def update_curve_info_message(curve_type):
    """Affiche un message d'aide selon le type de courbe sélectionné."""
    messages = {
        'random_walk': html.Div([
            html.Span('🎲 ', style={ 'fontSize': '16px' }),
            html.Span('Random Walk : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('Marche aléatoire. ', style={ 'color': '#888' }),
            html.Span('Bruit', style={ 'color': '#4CAF50' }),
            html.Span(' contrôle l\'amplitude des variations.', style={ 'color': '#888' }),
        ]),
        'trend': html.Div([
            html.Span('📈 ', style={ 'fontSize': '16px' }),
            html.Span('Trend : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('Tendance + bruit. ', style={ 'color': '#888' }),
            html.Span('Trend > 0', style={ 'color': '#4CAF50' }),
            html.Span(' = hausse, ', style={ 'color': '#888' }),
            html.Span('< 0', style={ 'color': '#f44336' }),
            html.Span(' = baisse.', style={ 'color': '#888' }),
        ]),
        'seasonal': html.Div([
            html.Span('🌊 ', style={ 'fontSize': '16px' }),
            html.Span('Seasonal : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('Cycle sinusoïdal journalier + bruit. ', style={ 'color': '#888' }),
            html.Span('Amplitude', style={ 'color': '#4CAF50' }),
            html.Span(' = force du cycle.', style={ 'color': '#888' }),
        ]),
        'lunch_effect': html.Div([
            html.Span('🍽️ ', style={ 'fontSize': '16px' }),
            html.Span('Lunch Effect : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('Baisse prix 12h-14h + bruit. ', style={ 'color': '#888' }),
            html.Span('Lunch effect', style={ 'color': '#4CAF50' }),
            html.Span(' = intensité de la baisse.', style={ 'color': '#888' }),
        ]),
        'sinusoidale': html.Div([
            html.Span('〰️ ', style={ 'fontSize': '16px' }),
            html.Span('Sinusoïdale : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('Oscillation régulière. ', style={ 'color': '#888' }),
            html.Span('Période', style={ 'color': '#4CAF50' }),
            html.Span(' = durée cycle, ', style={ 'color': '#888' }),
            html.Span('Bruit=0', style={ 'color': '#4CAF50' }),
            html.Span(' = parfait.', style={ 'color': '#888' }),
        ]),
        'plateau': html.Div([
            html.Span('📊 ', style={ 'fontSize': '16px' }),
            html.Span('Plateau : ', style={ 'color': '#FF8C00', 'fontWeight': 'bold' }),
            html.Span('N niveaux aléatoires répétés. ', style={ 'color': '#888' }),
            html.Span('Bruit=0', style={ 'color': '#4CAF50' }),
            html.Span(' = déterministe. ', style={ 'color': '#888' }),
            html.Span('Idéal pour tester l\'IA !', style={ 'color': '#2196F3', 'fontWeight': 'bold' }),
        ]),
    }
    return messages.get(curve_type, html.Div())


@app.callback(
    [
        Output('play_segments_graph', 'figure'),
        Output('play_df_store', 'data'),
        Output('play_train_backtest', 'disabled'),
        Output('play_model_ready', 'data'),
        Output('play_model_path', 'data'),
        Output('play_predictions_store', 'data'),
        Output('play_gen_summary', 'children'),
    ],
    [Input('play_generate', 'n_clicks')],
    [
        State('play_curve_type', 'value'),
        State('play_date_range', 'start_date'),
        State('play_date_range', 'end_date'),
        State('play_open_time', 'value'),
        State('play_close_time', 'value'),
        State('play_base_price', 'value'),
        State('play_noise', 'value'),
        State('play_trend_strength', 'value'),
        State('play_seasonality_amp', 'value'),
        State('play_sine_period', 'value'),
        State('play_nb_plateaux', 'value'),
        State('play_lunch_strength', 'value'),
        State('play_seed', 'value'),
    ],
    prevent_initial_call=True,
)
def generate_curve(n_clicks, curve_type, start_date, end_date, open_time, close_time, base_price, noise_val, trend_s, seas_amp, sine_period, nb_plateaux, lunch_s, seed):
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', paper_bgcolor='#000000', plot_bgcolor='#000000', font={ 'color': '#FFFFFF' }, title='Série synthétique — cliquer sur Générer', height=420, uirevision='play_segments')
    if not n_clicks:
        return empty_fig, None, True, dash.no_update, dash.no_update, None, html.Div("Cliquez sur Générer pour créer une nouvelle courbe.", style={ 'color': '#CCCCCC', 'fontSize': '12px' })
    try:
        df = generate_synthetic_timeseries(
            start_date, end_date,
            market_open=open_time or '09:30',
            market_close=close_time or '16:00',
            base_price=float(base_price or 100.0),
            data_type=str(curve_type or 'random_walk'),
            seed=int(seed) if seed is not None else None,
            noise=float(noise_val) if noise_val is not None else 0.0,
            trend_strength=float(trend_s) if trend_s is not None else 0.0,
            seasonality_amplitude=float(seas_amp) if seas_amp is not None else 0.0,
            lunch_effect_strength=float(lunch_s) if lunch_s is not None else 0.0,
            sine_period_minutes=int(sine_period) if sine_period is not None else 360,
            nb_plateaux=int(nb_plateaux) if nb_plateaux is not None else 3,
        )
        if df is None or df.empty:
            empty_fig.update_layout(title='Aucune donnée générée (plage/horaires vides)')
            return empty_fig, None, True, dash.no_update, dash.no_update, None, html.Div("Aucune donnée générée (plage/horaires vides)", style={ 'color': '#F59E0B', 'fontSize': '12px' })
        
        # Stocker les données
        store = df[['openPrice']].to_json(date_format='iso', orient='split')
        
        # Construire le graphe avec la courbe complète en orange vif
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['openPrice'], mode='lines', name='Prix',
            line={ 'color': '#FF8C00', 'width': 2 }
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' },
            title=f'📊 {curve_type.upper()} — {len(df)} points générés',
            height=420, uirevision='play_segments',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        msg = html.Div("Courbe générée. Modèle précédent toujours disponible pour la généralisation. Ré-entraîner si besoin.", style={ 'color': '#94a3b8', 'fontSize': '12px' })
        return fig, store, False, dash.no_update, dash.no_update, None, msg
    except Exception as e:
        empty_fig.update_layout(title=f'Erreur génération: {e}', height=420, uirevision='play_segments')
        return empty_fig, None, True, dash.no_update, dash.no_update, None, html.Div(f"Erreur génération: {e}", style={ 'color': '#EF4444', 'fontSize': '12px' })


@app.callback(
    [
        Output('play_saved_model', 'options'),
        Output('play_saved_model', 'value')
    ],
    [
        Input('play_symbol', 'value'),
        Input('play_saved_model_type_filter', 'value')
    ]
)
def populate_saved_models(symbol, model_type_filter):
    """Remplit la liste des modèles sauvegardés en fonction du symbole et du type."""
    options = []
    try:
        # Si un type est sélectionné (autre que 'all'), filtrer par type
        rows = []
        try:
            if model_type_filter and model_type_filter != 'all':
                rows = shM.list_models_by_type(model_type_filter)
            elif symbol:
                rows = shM.list_models_for_symbol(symbol)
            else:
                # Liste tous les modèles si aucun filtre
                rows = shM.list_models_by_type(None)
        except Exception as db_err:
            # Fallback si la colonne model_type n'existe pas dans la DB
            logging.warning(f"Error while listing models by type: {db_err}")
            try:
                rows = shM.list_models_for_symbol(symbol) if symbol else []
            except Exception:
                rows = []
        
        for row in rows:
            # Format: (id, date, trainScore, testScore, model_type, symbols)
            mid = row[0]
            date_val = row[1]
            train_s = row[2] if len(row) > 2 else None
            test_s = row[3] if len(row) > 3 else None
            m_type = row[4] if len(row) > 4 else 'lstm'
            symbols_json = row[5] if len(row) > 5 else None
            
            # Filtrer par symbole si spécifié
            if symbol:
                symbols_list = []
                if symbols_json:
                    import json
                    try:
                        symbols_list = json.loads(symbols_json) if isinstance(symbols_json, str) else symbols_json
                    except Exception:
                        symbols_list = []
                if symbols_list and symbol not in symbols_list:
                    continue
            
            # Emoji selon le type (utilise MODEL_TYPES factorisé)
            type_emoji = MODEL_TYPES.get(m_type, {}).get('icon', '❓')
            
            train_str = f"{train_s:.4f}" if train_s is not None else '-'
            test_str = f"{test_s:.4f}" if test_s is not None else '-'
            label = f"{type_emoji} {mid} — {str(date_val)[:10]} — train={train_str} test={test_str}"
            options.append({ 'label': label, 'value': mid })
        return options, (options[0]['value'] if options else None)
    except Exception:
        return [], None


def _build_lstm_model(look_back: int, num_features: int, nb_y: int, units: int, layers: int, lr: float, use_directional_accuracy: bool = True, prediction_type: str = 'return', loss_type: str = 'mse') -> tf.keras.Model:
    """
    Construit un modèle LSTM.
    
    loss_type:
    - 'mse': Mean Squared Error (défaut)
    - 'scaled_mse': MSE sur des targets multipliées par 100
    - 'mae': Mean Absolute Error
    """
    # Métrique de Directional Accuracy (DA)
    metrics_list = []
    if use_directional_accuracy:
        if prediction_type == 'price':
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur Prix normalisés: compare si le prix va au-dessus/en-dessous du prix de référence (1.0)
                true_dir = tf.sign(y_true - 1.0)
                pred_dir = tf.sign(y_pred - 1.0)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        else:
            def directional_accuracy_metric(y_true, y_pred):
                # DA sur retours: compare les signes des variations
                true_dir = tf.sign(y_true)
                pred_dir = tf.sign(y_pred)
                equal = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)
                return tf.reduce_mean(equal)
        
        try:
            directional_accuracy_metric.__name__ = 'directional_accuracy'
        except Exception:
            pass
        metrics_list.append(directional_accuracy_metric)
    
    # Choix de la loss
    if loss_type == 'mae':
        loss_fn = 'mae'
    else:
        loss_fn = 'mse'  # mse et scaled_mse utilisent tous deux mse (scaling fait sur les données)
    
    inputs = tf.keras.Input(shape=(int(look_back), int(num_features)))
    x = inputs
    for i in range(int(max(1, layers))):
        return_seq = (i != int(layers) - 1)
        x = tf.keras.layers.LSTM(int(units), return_sequences=return_seq, dropout=0.0)(x)
    
    if prediction_type == 'signal':
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        # Force loss/metrics pour signal
        loss_fn = 'sparse_categorical_crossentropy'
        metrics_list = ['accuracy']
    else:
        outputs = tf.keras.layers.Dense(int(nb_y))(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)), loss=loss_fn, metrics=metrics_list)
    return model


def _prepare_xy_from_store(store_json: str, look_back: int, stride: int, nb_y: int, first_minutes: int = None, prediction_type: str = 'return'):
    """
    Prépare les batches X et Y pour l'entraînement.
    prediction_type: 'return' (variation relative) ou 'price' (prix normalisé par rapport à la dernière observation)
    """
    if not store_json:
        return None, None, None, None, 0
    df = pd.read_json(StringIO(store_json), orient='split')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    nb_per_day = estimate_nb_quotes_per_day(df)
    if nb_per_day <= 0:
        return None, None, None, None, 0
    # Split train/test par jours (80/20)
    days = df.index.normalize().unique()
    if len(days) < 2:
        split = len(df)
        train_df = df.iloc[:split]
        test_df = df.iloc[0:0]
    else:
        split_idx = int(len(days) * 0.8)
        split_day = days[split_idx - 1]
        train_df = df.loc[df.index.normalize() <= split_day]
        test_df = df.loc[df.index.normalize() > split_day]

    obs_window = int(first_minutes) if first_minutes is not None and first_minutes > 0 else int(look_back * stride)

    def create_xy(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        if dataset is None or dataset.empty:
            return np.zeros((0, look_back, 1), dtype=float), np.zeros((0, nb_y), dtype=float)
        # Itérer par jour
        norm = dataset.index.normalize()
        days_u = norm.unique()
        for d in days_u:
            day_df = dataset.loc[norm == d, ['openPrice']]
            if day_df.shape[0] < obs_window + max(2, nb_y):
                continue
            # Sélectionner les obs_window premières minutes pour construire la fenêtre d'entrée
            if obs_window < look_back * stride:
                available_points = min(obs_window, day_df.shape[0])
                if available_points < look_back:
                    continue
                step = max(1, available_points // look_back)
                seq = day_df.iloc[0: available_points: step].iloc[:look_back].to_numpy(dtype=float)
            else:
                step = max(1, obs_window // look_back)
                seq = day_df.iloc[0: obs_window: step].iloc[:look_back].to_numpy(dtype=float)
            
            if seq.shape[0] != look_back:
                continue
            
            base_price = seq[-1, 0]
            if base_price == 0:
                continue
            # Normaliser input
            seq[:, 0] = seq[:, 0] / base_price
            if seq.shape[1] >= 2:
                seq[:, 1] = np.log1p(np.clip(seq[:, 1], a_min=0.0, a_max=None))
            
            remainder = day_df.shape[0] - obs_window
            if remainder <= 0:
                continue
            if remainder <= nb_y:
                continue
            stride_y = remainder // (nb_y + 1)
            if stride_y == 0:
                continue
            offsets = [(j + 1) * stride_y for j in range(nb_y)]
            
            y_vals = []
            prev_price = base_price
            prices_list = [base_price]  # Pour logging
            
            if prediction_type == 'signal':
                # Classification 5 classes : -2, -1, 0, 1, 2 (mappées 0..4)
                # Horizon = dernier offset
                horizon = offsets[-1] if offsets else 1
                if obs_window + horizon < day_df.shape[0]:
                    final_p = float(day_df.iloc[obs_window + horizon, 0])
                    ret = (final_p - base_price) / base_price
                    
                    # Seuils arbitraires (à raffiner)
                    if ret < -0.005: label = 0   # -2 (Strong Drop)
                    elif ret < -0.001: label = 1 # -1 (Drop)
                    elif ret < 0.001: label = 2  # 0 (Flat)
                    elif ret < 0.005: label = 3  # 1 (Rise)
                    else: label = 4              # 2 (Strong Rise)
                    y_vals.append(label)
                    prices_list.append(final_p)
            else:
                for i, off in enumerate(offsets):
                    y_price = float(day_df.iloc[obs_window + off, 0])
                    prices_list.append(y_price)
                    
                    if prediction_type == 'price':
                        # Mode Prix : ratio par rapport au dernier prix connu (base_price)
                        val = y_price / base_price
                        y_vals.append(val)
                    else:
                        # Mode Return (défaut) : variations relatives pas à pas
                        if i == 0:
                            variation = (y_price / prev_price) - 1.0
                            y_vals.append(variation)
                        else:
                            prev_off = offsets[i - 1]
                            prev_price_iter = float(day_df.iloc[obs_window + prev_off, 0])
                        variation = (y_price / prev_price_iter) - 1.0
                        y_vals.append(variation)
                    prev_price = y_price
            
            # Log détaillé seulement pour le premier
            if len(X) == 0:
                logging.info(f"[Prepare XY] Mode={prediction_type}. Exemple premier échantillon:")
                logging.info(f"  Prix: {[f'{p:.2f}' for p in prices_list]}")
                logging.info(f"  Valeurs cibles: {[f'{v:.4f}' for v in y_vals]}")
            
            X.append(seq)
            Y.append(y_vals)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        return X, Y

    trainX, trainY = create_xy(train_df)
    testX, testY = create_xy(test_df)
    return trainX, trainY, testX, testY, nb_per_day


def _prepare_xy_for_inference(store_json: str, look_back: int, stride: int, nb_y: int, first_minutes: int = None, prediction_type: str = 'return'):
    """
    Prépare X/Y pour l'inférence (généralisation) sur la courbe courante.
    Contrairement à _prepare_xy_from_store, on ne fait pas de split train/test :
    chaque jour valide fournit un seul échantillon (X, Y) basé sur les premières minutes.
    """
    if not store_json:
        return None, None, None, 0, []
    df = pd.read_json(StringIO(store_json), orient='split')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    if df is None or df.empty:
        return None, None, None, 0, []

    obs_window = int(first_minutes) if first_minutes is not None and first_minutes > 0 else int(look_back * stride)

    norm = df.index.normalize()
    days_u = norm.unique()

    X_list = []
    Y_list = []
    sample_days = []

    for d in days_u:
        day_df = df.loc[norm == d, ['openPrice']]
        if day_df.shape[0] < obs_window + max(2, nb_y):
            continue

        # Sélection des points d'observation (mêmes règles que _prepare_xy_from_store)
        if obs_window < look_back * stride:
            available_points = min(obs_window, day_df.shape[0])
            if available_points < look_back:
                continue
            step = max(1, available_points // look_back)
            seq = day_df.iloc[0: available_points: step].iloc[:look_back].to_numpy(dtype=float)
        else:
            step = max(1, obs_window // look_back)
            seq = day_df.iloc[0: obs_window: step].iloc[:look_back].to_numpy(dtype=float)

        if seq.shape[0] != look_back:
            continue

        base_price = seq[-1, 0]
        if base_price == 0:
            continue

        # Normaliser input
        seq[:, 0] = seq[:, 0] / base_price

        remainder = day_df.shape[0] - obs_window
        if remainder <= nb_y:
            continue

        stride_y = remainder // (nb_y + 1)
        if stride_y <= 0:
            continue

        offsets = [(j + 1) * stride_y for j in range(nb_y)]

        y_vals = []
        prev_price = base_price

        for i, off in enumerate(offsets):
            y_price = float(day_df.iloc[obs_window + off, 0])

            if prediction_type == 'price':
                # Mode Prix : ratio par rapport au dernier prix connu (base_price)
                val = y_price / base_price
                y_vals.append(val)
            else:
                # Mode Return : variations relatives pas à pas
                if i == 0:
                    variation = (y_price / prev_price) - 1.0
                    y_vals.append(variation)
                else:
                    prev_off = offsets[i - 1]
                    prev_price_iter = float(day_df.iloc[obs_window + prev_off, 0])
                    variation = (y_price / prev_price_iter) - 1.0
                    y_vals.append(variation)
                prev_price = y_price

        X_list.append(seq)
        Y_list.append(y_vals)
        sample_days.append(d)

    if not X_list:
        return None, None, df, obs_window, []

    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    return X, Y, df, obs_window, sample_days


def _build_segments_graph_from_store(store_json: str, look_back: int, stride: int, first_minutes: int, predictions=None, nb_y: int = None, predictions_train=None, prediction_type: str = 'return', extra_predictions=None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments entraînement / test', height=320, uirevision='play_segments')
    if not store_json:
        return fig
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
    except Exception:
        return fig
    if df is None or df.empty:
        return fig
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(how='any')
    except Exception:
        pass
    if df.index.dtype == object:
        return fig
    idx = df.index
    norm = idx.normalize()
    days = norm.unique()
    if len(days) == 0:
        return fig
    try:
        fig.add_trace(go.Scatter(x=idx, y=df['openPrice'].values, mode='lines', name='Série', line={ 'color': '#888888', 'width': 1 }, opacity=0.35))
    except Exception:
        pass
    split_idx = int(len(days) * 0.8)
    split_day = days[split_idx - 1] if split_idx > 0 else days[0]
    n = len(df)
    masks = {
        'train_obs': np.zeros(n, dtype=bool),
        'train_rest': np.zeros(n, dtype=bool),
        'test_obs': np.zeros(n, dtype=bool),
        'test_rest': np.zeros(n, dtype=bool),
    }
    obs_len_steps = int(max(1, first_minutes or 60))
    for d in days:
        day_mask = (norm == d)
        pos = np.where(day_mask)[0]
        if pos.size == 0:
            continue
        obs_len = min(obs_len_steps, pos.size)
        obs_idx = pos[:obs_len]
        rest_idx = pos[obs_len:]
        if d <= split_day:
            masks['train_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['train_rest'][rest_idx] = True
        else:
            masks['test_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['test_rest'][rest_idx] = True
    def add_series(name, mask, color, width=2):
        y = np.where(mask, df['openPrice'].values, np.nan)
        fig.add_trace(go.Scatter(x=idx, y=y, mode='lines', name=name, line={ 'color': color, 'width': width }))
    add_series('Train (premières min)', masks['train_obs'], '#1f77b4', 2)
    add_series('Train (reste)', masks['train_rest'], '#2ca02c', 2)
    add_series('Test (premières min)', masks['test_obs'], '#9467bd', 2)
    add_series('Test (reste)', masks['test_rest'], '#d62728', 2)

    # Zone de couleur de fond pour le mode SIGNAL
    if prediction_type == 'signal':
        # On va créer des shapes (rectangles) pour chaque prédiction
        shapes = []
        
        def add_signal_shapes(preds, day_list):
            if preds is None: return
            preds_arr = np.array(preds).flatten()
            pred_idx = 0
            
            for day in day_list:
                day_mask = (norm == day)
                pos = np.where(day_mask)[0]
                if pos.size == 0: continue
                obs_len = min(obs_len_steps, pos.size)
                if obs_len >= pos.size: continue
                
                rest_pos = pos[obs_len:]
                if len(rest_pos) == 0: continue
                
                # Le signal s'applique à toute la période future "nb_y" (ou tout le reste du jour)
                if pred_idx >= len(preds_arr): break
                
                signal_val = int(preds_arr[pred_idx])
                pred_idx += 1
                
                # Mapping couleur
                # 0: -2 (Strong Drop) -> Rouge saturé
                # 1: -1 (Drop) -> Rouge pâle
                # 2: 0 (Flat) -> Transparent/Gris
                # 3: +1 (Rise) -> Vert pâle
                # 4: +2 (Strong Rise) -> Vert saturé
                color = None
                if signal_val == 0: color = 'rgba(255, 0, 0, 0.4)'
                elif signal_val == 1: color = 'rgba(255, 100, 100, 0.2)'
                elif signal_val == 3: color = 'rgba(100, 255, 100, 0.2)'
                elif signal_val == 4: color = 'rgba(0, 255, 0, 0.4)'
                
                if color:
                    # Début et fin de la zone prédite
                    x0 = idx[rest_pos[0]]
                    x1 = idx[rest_pos[-1]] # Jusqu'à la fin du jour ou horizon
                    
                    shapes.append(dict(
                        type="rect",
                        xref="x", yref="paper",
                        x0=x0, y0=0, x1=x1, y1=1,
                        fillcolor=color,
                        opacity=0.5,
                        layer="below",
                        line_width=0,
                    ))

        train_days = days[:split_idx]
        test_days = days[split_idx:]
        
        # Pour le signal, on utilise 'predictions_train' et 'predictions' (qui contiennent les classes 0..4)
        add_signal_shapes(predictions_train, train_days)
        add_signal_shapes(predictions, test_days)
        
        fig.update_layout(shapes=shapes)

    def reconstruct_predictions(predictions_data, day_list, color_name, color_hex):
        if predictions_data is not None and len(predictions_data) > 0:
            try:
                pred_idx_flat = []
                pred_values_flat = []
                preds_array = np.array(predictions_data) if isinstance(predictions_data, list) else predictions_data
                preds_flat = preds_array.flatten()
                
                pred_idx_in_flat = 0
                for day_idx, day in enumerate(day_list):
                    day_mask = (norm == day)
                    pos = np.where(day_mask)[0]
                    if pos.size == 0: continue
                    obs_len = min(obs_len_steps, pos.size)
                    if obs_len >= pos.size: continue
                    rest_pos = pos[obs_len:]
                    if len(rest_pos) == 0: continue
                    
                    remainder = len(rest_pos)
                    # Utiliser le nb_y passé en paramètre s'il est valide, sinon fallback
                    nb_y_used = nb_y if nb_y is not None and nb_y > 0 else min(5, remainder)
                    
                    remaining_preds = len(preds_flat) - pred_idx_in_flat
                    if remaining_preds < nb_y_used:
                        nb_y_used = remaining_preds
                    if nb_y_used <= 0: continue
                    
                    base_price = float(df.iloc[pos[obs_len - 1]]['openPrice'])
                    if base_price == 0: continue
                    
                    stride_y = remainder // (nb_y_used + 1) if nb_y_used > 0 else 1
                    offsets = [(j + 1) * stride_y for j in range(min(nb_y_used, remainder))]
                    
                    current_pred_price = base_price
                    for i in range(nb_y_used):
                        if pred_idx_in_flat >= len(preds_flat): break
                        if i < len(offsets) and offsets[i] < len(rest_pos):
                            off = offsets[i]
                            pred_val = float(preds_flat[pred_idx_in_flat])
                            
                            if prediction_type == 'price':
                                # Mode Prix : pred_val est un ratio par rapport à base_price
                                current_pred_price = base_price * pred_val
                            elif prediction_type == 'signal':
                                # Mode Signal : pred_val est une classe 0..4
                                # On visualise par un décalage artificiel pour voir la décision
                                cls = int(pred_val)
                                # 0->-2%, 1->-1%, 2->0%, 3->+1%, 4->+2% (visuel)
                                current_pred_price = base_price * (1.0 + (cls - 2) * 0.01)
                            else:
                                # Mode Return : pred_val est une variation relative pas à pas
                                current_pred_price = current_pred_price * (1.0 + pred_val)
                                
                            pred_idx_flat.append(idx[rest_pos[off]])
                            pred_values_flat.append(current_pred_price)
                            pred_idx_in_flat += 1
                
                if pred_values_flat:
                    fig.add_trace(go.Scatter(x=pred_idx_flat, y=pred_values_flat, mode='lines+markers', name=color_name, line={ 'color': color_hex, 'width': 2 }, marker={ 'size': 4 }))
            except Exception:
                pass

    train_days = days[:split_idx]
    test_days = days[split_idx:]
    reconstruct_predictions(predictions_train, train_days, 'Prédiction (train)', '#17becf')
    reconstruct_predictions(predictions, test_days, 'Prédiction (test)', '#FF8C00')
    
    if extra_predictions:
        for extra in extra_predictions:
            # extra is dict: {'test': flat_pred, 'train': flat_pred, 'name': str, 'color': str}
            if 'train' in extra:
                 reconstruct_predictions(extra['train'], train_days, f"{extra['name']} (train)", extra.get('color', '#888888'))
            if 'test' in extra:
                 reconstruct_predictions(extra['test'], test_days, f"{extra['name']} (test)", extra.get('color', '#FF8C00'))
    
    return fig


def _build_generalization_figure(df: pd.DataFrame, sample_days: list, obs_window: int, y_pred: np.ndarray, nb_y: int, prediction_type: str = 'return') -> tuple[go.Figure, float, float, int]:
    """
    Construit un graphe pour visualiser la généralisation du modèle sur la courbe courante.
    Affiche la série réelle + quelques points futurs réels vs prédits.
    Retourne (figure, MAE, RMSE, nb_points).
    """
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={ 'color': '#FFFFFF' },
        title='Test de généralisation — courbe actuelle',
        height=420,
        uirevision='play_generalization'
    )

    if df is None or df.empty or y_pred is None or len(sample_days) == 0:
        return fig, 0.0, 0.0, 0

    idx = df.index
    norm = idx.normalize()

    # Trace de la série complète en fond (un peu plus visible, couleur orange)
    try:
        fig.add_trace(go.Scatter(
            x=idx,
            y=df['openPrice'].values,
            mode='lines',
            name='Série (nouvelle)',
            line={ 'color': '#FF8C00', 'width': 1.8 },
            opacity=0.55
        ))
    except Exception:
        pass

    all_x = []
    all_true = []
    all_pred = []

    for i, d in enumerate(sample_days):
        if i >= y_pred.shape[0]:
            break

        day_mask = (norm == d)
        pos = np.where(day_mask)[0]
        if pos.size == 0:
            continue

        obs_len = min(int(obs_window), pos.size)
        if obs_len >= pos.size:
            continue

        rest_pos = pos[obs_len:]
        remainder = len(rest_pos)
        if remainder <= nb_y:
            continue

        stride_y = remainder // (nb_y + 1)
        if stride_y <= 0:
            continue

        offsets = [(j + 1) * stride_y for j in range(nb_y)]

        base_index = pos[obs_len - 1]
        base_price = float(df.iloc[base_index]['openPrice'])
        if base_price == 0:
            continue

        current_pred_price = base_price
        y_pred_vec = np.array(y_pred[i]).flatten()

        for j, off in enumerate(offsets):
            if j >= len(y_pred_vec):
                break
            if off >= len(rest_pos):
                break

            true_index = rest_pos[off]
            true_price = float(df.iloc[true_index]['openPrice'])
            pred_val = float(y_pred_vec[j])

            if prediction_type == 'price':
                current_pred_price = base_price * pred_val
            else:
                current_pred_price = current_pred_price * (1.0 + pred_val)

            all_x.append(idx[true_index])
            all_true.append(true_price)
            all_pred.append(current_pred_price)

    if not all_x:
        return fig, 0.0, 0.0, 0

    # Trier par temps
    # order = np.argsort(np.array(all_x, dtype='datetime64[ns]'))
    # x_sorted = [all_x[k] for k in order]
    # true_sorted = np.array(all_true)[order]
    # pred_sorted = np.array(all_pred)[order]
    
    # Ne PAS trier globalement par temps pour éviter de relier la fin d'un jour au début du suivant
    # On garde l'ordre d'insertion (jour par jour) et on utilise 'markers' pour ne pas avoir de traits de liaison bizarres
    x_sorted = all_x
    true_sorted = np.array(all_true)
    pred_sorted = np.array(all_pred)

    # Courbe prédite
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=pred_sorted,
        mode='markers',
        name='Prix prédit (généralisation)',
        marker={ 'size': 5, 'color': '#00E0FF', 'symbol': 'diamond' }
    ))

    errors = pred_sorted - true_sorted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    n_points = int(len(errors))

    return fig, mae, rmse, n_points


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
    ],
    [
        # Déclenché uniquement par les changements de paramètres, pas par le store
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
        Input('play_first_minutes', 'value'),
        Input('play_nb_y', 'value'),
        Input('play_prediction_type', 'value'),
    ],
    [
        State('play_df_store', 'data'),  # Utiliser State au lieu de Input
    ],
    prevent_initial_call=True,
)
def update_segments_graph(look_back, stride, first_minutes, nb_y, prediction_type, store_json):
    """Met à jour le graphe des segments quand les paramètres changent (pas quand la courbe est générée)."""
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Générer une courbe d\'abord', height=420, uirevision='play_segments')
    
    if not store_json:
        return (empty_fig,)
    try:
        look_back_val = int(look_back or DEFAULT_LOOK_BACK)
        stride_val = int(stride or DEFAULT_STRIDE)
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
        nb_y_val = int(nb_y or DEFAULT_NB_Y)
        pred_type = prediction_type or 'return'
        fig = _build_segments_graph_from_store(store_json, look_back_val, stride_val, first_minutes_val, None, nb_y_val, None, pred_type)
        return (fig,)
    except Exception:
        return (empty_fig,)


@app.callback(
    [
        Output('play_nb_y', 'max'),
        Output('play_nb_y', 'value'),
        Output('play_nb_y', 'marks'),
        Output('play_nb_y_value', 'children'),
    ],
    [
        Input('play_first_minutes', 'value'),
        Input('play_open_time', 'value'),
        Input('play_close_time', 'value'),
    ],
    [
        State('play_nb_y', 'value'),
    ]
)
def update_nb_y_slider(first_minutes, open_time, close_time, current_nb_y):
    # Defaults
    try:
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
    except Exception:
        first_minutes_val = DEFAULT_FIRST_MINUTES
    # Parse times HH:MM
    def parse_minutes(hhmm, fallback):
        try:
            parts = str(hhmm or '').split(':')
            h = int(parts[0]); m = int(parts[1])
            return h * 60 + m
        except Exception:
            return fallback
    open_min = parse_minutes(open_time, 9 * 60 + 30)   # 09:30
    close_min = parse_minutes(close_time, 16 * 60)     # 16:00
    day_len = max(0, close_min - open_min)
    remainder = max(0, day_len - max(0, first_minutes_val))
    # Points répartis uniformément: il faut au moins nb_y+1 minutes pour répartir nb_y points
    # Donc max_nb_y = remainder - 1 (minimum), mais on veut au moins 1 point minimum
    max_nb_y = max(1, max(0, remainder - 1))
    try:
        cur_val = int(current_nb_y or 5)
    except Exception:
        cur_val = 5
    new_val = min(max_nb_y, max(1, cur_val))
    # Marks simples (début/fin) pour performance
    marks = { 1: '1', max_nb_y: str(max_nb_y) }
    value_display = html.Span(f"Valeur actuelle: {new_val}", style={ 'fontWeight': 'bold' })
    logging.info(f"[UI] Ajustement slider nb_y — day_len={day_len} remainder={remainder} max={max_nb_y} value={new_val}")
    return max_nb_y, new_val, marks, value_display

@app.callback(
    Output('play_nb_y_value', 'children'),
    Input('play_nb_y', 'value'),
)
def update_nb_y_display(nb_y_value):
    """Met à jour l'affichage de la valeur du slider nb_y quand l'utilisateur le bouge"""
    try:
        val = int(nb_y_value or 5)
    except Exception:
        val = 5
    return html.Span(f"Valeur actuelle: {val}", style={ 'fontWeight': 'bold' })

@app.callback(
    Output('play_first_minutes', 'value'),
    [
        Input('play_look_back', 'value'),
        Input('play_stride', 'value'),
    ],
    [State('play_first_minutes', 'value')],
    prevent_initial_call=True
)
def adjust_first_minutes(look_back, stride, current_first_minutes):
    """
    Ajuste automatiquement le paramètre 'play_first_minutes' pour respecter la contrainte :
    first_minutes >= look_back * stride
    """
    try:
        # Parse look_back string
        look_back_str = str(look_back or DEFAULT_LOOK_BACK)
        window_sizes = []
        for x in look_back_str.split(','):
            x = x.strip()
            if x.isdigit():
                window_sizes.append(int(x))
        if not window_sizes: window_sizes = [DEFAULT_LOOK_BACK]
        
        max_lb = max(window_sizes)
        st = int(stride or DEFAULT_STRIDE)
        fm = int(current_first_minutes or DEFAULT_FIRST_MINUTES)
        
        min_required = max_lb * st
        
        if fm < min_required:
            logging.info(f"[UI] Auto-adjust: first_minutes ({fm}) < max_look_back*stride ({min_required}). Updating to {min_required}.")
            return min_required
        
        return dash.no_update
    except Exception:
        return dash.no_update


@app.callback(
    Output('play_mini_preview', 'figure'),
    [Input('play_df_store', 'data')],
    [State('play_curve_type', 'value')]
)
def update_mini_preview(store_json, curve_type):
    if not store_json:
        return {'data': [], 'layout': {'height': 1, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}}
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
        if df.empty: return {'data': [], 'layout': {'height': 1}}
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['openPrice'], mode='lines', 
            line={'color': '#FF8C00', 'width': 1.5},
            hoverinfo='none'
        ))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=150,
            showlegend=False
        )
        return fig
    except Exception:
        return {'data': [], 'layout': {'height': 1}}


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
        Output('play_predictions_store', 'data'),
        Output('play_run_backtest', 'disabled'),
        Output('play_model_ready', 'data'),
        Output('play_model_path', 'data'),
    ],
    [Input('play_train_backtest', 'n_clicks')],
    [
        State('play_df_store', 'data'),
        State('play_look_back', 'value'),
        State('play_stride', 'value'),
        State('play_nb_y', 'value'),
        State('play_first_minutes', 'value'),
        State('play_use_directional_accuracy', 'value'),
        State('play_loss_type', 'value'),
        State('play_units', 'value'),
        State('play_layers', 'value'),
        State('play_lr', 'value'),
        State('play_epochs', 'value'),
        State('play_prediction_type', 'value'),
        # Type de modèle
        State('play_model_type', 'value'),
        # Paramètres Transformer
        State('play_embed_dim', 'value'),
        State('play_num_heads', 'value'),
        State('play_transformer_layers', 'value'),
        State('play_ff_multiplier', 'value'),
        State('play_dropout', 'value'),
        # Paramètres Hybride
        State('play_hybrid_lstm_units', 'value'),
        State('play_hybrid_lstm_layers', 'value'),
        State('play_hybrid_embed_dim', 'value'),
        State('play_hybrid_num_heads', 'value'),
        State('play_hybrid_trans_layers', 'value'),
        State('play_fusion_mode', 'value'),
        State('play_hybrid_dropout', 'value'),
    ],
    background=True,
    progress=[
        Output('play_training_progress', 'children'),
        Output('play_training_history', 'figure'),
    ],
    running=[(Output('play_train_backtest', 'disabled'), True, False)],
)
def train_model(
    set_progress, n_clicks, store_json, look_back, stride, nb_y, first_minutes, 
    use_directional_accuracy, loss_type, units, layers, lr, epochs, prediction_type,
    model_type, embed_dim, num_heads, transformer_layers, ff_multiplier, dropout,
    hybrid_lstm_units, hybrid_lstm_layers, hybrid_embed_dim, hybrid_num_heads, hybrid_trans_layers, fusion_mode, hybrid_dropout
):
    
    global play_last_model, play_last_model_meta
    history_fig = go.Figure()
    history_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='En attente...', height=300, uirevision='play_hist')
    empty_seg_fig = go.Figure()
    empty_seg_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Segments — en attente', height=420, uirevision='play_segments')
    
    if not n_clicks:
        return empty_seg_fig, None, True, False, None
    try:
        # Forcer CPU si les drivers GPU posent problème
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                set_progress((html.Div(f"GPU détectés: {len(gpus)} — désactivation pour ce Playground"), history_fig))
                tf.config.set_visible_devices([], 'GPU')
            else:
                set_progress((html.Div("Aucun GPU détecté — utilisation CPU"), history_fig))
        except Exception:
            set_progress((html.Div("Impossible de configurer les devices GPU — fallback CPU"), history_fig))
        
        # Parsing des fenêtres (look_back peut être "60, 120")
        try:
            look_back_str = str(look_back or DEFAULT_LOOK_BACK)
            window_sizes = []
            for x in look_back_str.split(','):
                x = x.strip()
                if x.isdigit():
                    window_sizes.append(int(x))
            if not window_sizes:
                window_sizes = [DEFAULT_LOOK_BACK]
            window_sizes = sorted(list(set(window_sizes)))
        except Exception:
            window_sizes = [DEFAULT_LOOK_BACK]

        extra_predictions = []
        final_predictions_data = None
        final_model_path = None
        predictions_flat_main = None
        predictions_train_flat_main = None
        da_main = None
        
        # Couleurs pour les modèles supplémentaires
        colors = ['#FF00FF', '#FFFF00', '#00FF00', '#00E0FF', '#FF8C00'] # Cycle de couleurs

        stride_val = int(stride or DEFAULT_STRIDE)
        nb_y_val = int(nb_y or DEFAULT_NB_Y)
        first_minutes_val = int(first_minutes or DEFAULT_FIRST_MINUTES)
        units_val = int(units or DEFAULT_UNITS)
        layers_val = int(layers or DEFAULT_LAYERS)
        lr_val = float(lr or DEFAULT_LEARNING_RATE)
        pred_type = prediction_type or 'return'
        loss_type_val = loss_type or 'mse'
        model_type_val = model_type or 'lstm'
        use_da = use_directional_accuracy if use_directional_accuracy is not None else True
        
        # Callback de progression - historique métriques
        accs, vaccs, losses, vlosses = [], [], [], []
        
        def _make_hist_fig():
            # Figure avec deux axes Y: gauche = Loss (échelle log), droite = DA (0..100%)
            fig_h = go.Figure()
            
            # Traces Loss (axe gauche)
            if losses:
                fig_h.add_trace(go.Scatter(
                    x=list(range(1, len(losses)+1)), y=losses, 
                    mode='lines+markers', name='Loss train', 
                    line={ 'color': '#2ca02c', 'width': 2 }, 
                    marker={ 'size': 6 },
                    yaxis='y'
                ))
            if vlosses:
                fig_h.add_trace(go.Scatter(
                    x=list(range(1, len(vlosses)+1)), y=vlosses, 
                    mode='lines+markers', name='Loss val', 
                    line={ 'color': '#d62728', 'width': 2 }, 
                    marker={ 'size': 6 },
                    yaxis='y'
                ))
            
            # Traces DA (axe droit)
            if accs:
                accs_pct = [a * 100 for a in accs]
                fig_h.add_trace(go.Scatter(
                    x=list(range(1, len(accs_pct)+1)), y=accs_pct, 
                    mode='lines+markers', name='DA train %', 
                    line={ 'color': '#1f77b4', 'width': 2, 'dash': 'dot' }, 
                    marker={ 'size': 6 },
                    yaxis='y2'
                ))
            if vaccs:
                vaccs_pct = [a * 100 for a in vaccs]
                fig_h.add_trace(go.Scatter(
                    x=list(range(1, len(vaccs_pct)+1)), y=vaccs_pct, 
                    mode='lines+markers', name='DA val %', 
                    line={ 'color': '#ff7f0e', 'width': 2, 'dash': 'dot' }, 
                    marker={ 'size': 6 },
                    yaxis='y2'
                ))
            
            y_cfg = { 'title': 'Loss', 'side': 'left', 'type': 'log' }
            loss_info = ''
            if losses or vlosses:
                all_loss = [l for l in (losses + vlosses) if l is not None and l > 0]
                if all_loss:
                    current_loss = float(all_loss[-1])
                    if current_loss < 0.001:
                        loss_info = f' (actuel: {current_loss:.2e})'
                    else:
                        loss_info = f' (actuel: {current_loss:.6f})'
            
            title_text = f'📊 Loss{loss_info} & DA'
            
            fig_h.update_layout(
                template='plotly_dark',
                paper_bgcolor='#000000',
                plot_bgcolor='#000000',
                font={ 'color': '#FFFFFF' },
                title=title_text,
                height=300,
                uirevision='play_hist',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font={'size': 10}),
                margin=dict(t=60, b=40, l=60, r=60),
                yaxis=y_cfg,
                yaxis2={ 
                    'title': 'DA %', 
                    'overlaying': 'y', 
                    'side': 'right', 
                    'range': [0, 100],
                    'ticksuffix': '%'
                }
            )
            return fig_h

        for i, look_back_val in enumerate(window_sizes):
            is_last = (i == len(window_sizes) - 1)
            msg_prefix = f"[Modèle {i+1}/{len(window_sizes)} Win={look_back_val}] "
            set_progress((html.Div(f'{msg_prefix}Préparation des données...'), history_fig))
            
            logging.info(f"{msg_prefix}Paramètres batch: look_back={look_back_val}, type={pred_type}")
            trainX, trainY, testX, testY, nb_per_day = _prepare_xy_from_store(store_json, look_back_val, stride_val, nb_y_val, first_minutes_val, pred_type)
            
            if trainX is None or trainX.shape[0] == 0:
                logging.warning(f"{msg_prefix}Pas de données, skipping.")
                continue

            num_features = trainX.shape[-1]
            scale_factor = 1.0
            if loss_type_val == 'scaled_mse':
                scale_factor = 100.0
                trainY = trainY * scale_factor
                if testY is not None: testY = testY * scale_factor
            
            set_progress((html.Div(f'{msg_prefix}Construction du modèle ({model_type_val})...'), history_fig))
            
            if model_type_val == 'transformer' and TRANSFORMER_AVAILABLE:
                embed_dim_val = int(embed_dim or DEFAULT_EMBED_DIM)
                num_heads_val = int(num_heads or DEFAULT_NUM_HEADS)
                trans_layers_val = int(transformer_layers or DEFAULT_TRANSFORMER_LAYERS)
                ff_mult_val = int(ff_multiplier or DEFAULT_FF_MULTIPLIER)
                dropout_val = float(dropout or DEFAULT_DROPOUT)
                model = create_transformer_model(look_back_val, int(num_features), nb_y_val, embed_dim_val, num_heads_val, trans_layers_val, ff_mult_val, dropout_val, lr_val, use_da, pred_type)
            elif model_type_val == 'hybrid' and TRANSFORMER_AVAILABLE:
                h_lstm_units = int(hybrid_lstm_units or DEFAULT_UNITS)
                h_lstm_layers = int(hybrid_lstm_layers or DEFAULT_LAYERS)
                h_embed_dim = int(hybrid_embed_dim or DEFAULT_EMBED_DIM)
                h_num_heads = int(hybrid_num_heads or DEFAULT_NUM_HEADS)
                h_trans_layers = int(hybrid_trans_layers or 1)
                h_fusion = fusion_mode or DEFAULT_FUSION_MODE
                h_dropout = float(hybrid_dropout or DEFAULT_DROPOUT)
                model = create_hybrid_lstm_transformer_model(look_back_val, int(num_features), nb_y_val, h_lstm_units, h_lstm_layers, h_embed_dim, h_num_heads, h_trans_layers, DEFAULT_FF_MULTIPLIER, h_dropout, lr_val, use_da, pred_type, h_fusion)
            else:
                model = _build_lstm_model(look_back_val, int(num_features), nb_y_val, units_val, layers_val, lr_val, use_da, pred_type, loss_type_val)
            
            # Entraînement
            num_epochs = int(epochs or DEFAULT_EPOCHS)
            set_progress((html.Div(f'{msg_prefix}Entraînement ({num_epochs} epochs)...'), history_fig))
            
            # Callback complet pour mettre à jour history_fig
            class FullProgCB(tf.keras.callbacks.Callback):
                def __init__(self, total_epochs, metric_losses, metric_vlosses, metric_accs, metric_vaccs):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.losses = metric_losses
                    self.vlosses = metric_vlosses
                    self.accs = metric_accs
                    self.vaccs = metric_vaccs
                    self.last_update = time.time()
                
                def on_epoch_end(self, epoch, logs=None):
                    # Collect metrics
                    l = logs.get('loss')
                    vl = logs.get('val_loss')
                    a = logs.get('accuracy') or logs.get('directional_accuracy') 
                    va = logs.get('val_accuracy') or logs.get('val_directional_accuracy')

                    if l is not None: self.losses.append(float(l))
                    if vl is not None: self.vlosses.append(float(vl))
                    if a is not None: self.accs.append(float(a))
                    if va is not None: self.vaccs.append(float(va))
                    
                    # Update graph every 0.5s or last epoch
                    if (time.time() - self.last_update > 0.5) or (epoch == self.total_epochs - 1):
                        self.last_update = time.time()
                        new_fig = _make_hist_fig()
                        loss_txt = f"{l:.4f}" if l else "?"
                        set_progress((html.Div(f"{msg_prefix}Epoch {epoch+1}/{self.total_epochs} - Loss={loss_txt}"), new_fig))

            model.fit(trainX, trainY, epochs=num_epochs, validation_data=(testX, testY) if (testX is not None and testX.size) else None, verbose=0, callbacks=[FullProgCB(num_epochs, losses, vlosses, accs, vaccs)])
            
            # Prédictions
            y_pred = model.predict(testX, verbose=0) if (testX is not None and testX.size) else None
            y_pred_train = model.predict(trainX, verbose=0) if (trainX is not None and trainX.size) else None
            
            if pred_type == 'signal':
                if y_pred is not None: 
                    y_pred = np.argmax(y_pred, axis=1).reshape(-1, 1)
                if y_pred_train is not None:
                    y_pred_train = np.argmax(y_pred_train, axis=1).reshape(-1, 1)

            if scale_factor != 1.0:
                if y_pred is not None: y_pred = y_pred / scale_factor
                if y_pred_train is not None: y_pred_train = y_pred_train / scale_factor
                if testY is not None: testY = testY / scale_factor

            preds_flat = y_pred.flatten().tolist() if y_pred is not None else []
            preds_train_flat = y_pred_train.flatten().tolist() if y_pred_train is not None else []
            
            # Sauvegarde modèle (toujours écraser last_model pour le dernier)
            play_last_model = model
            play_last_model_meta = {
                'look_back': look_back_val,
                'stride': stride_val,
                'nb_y': nb_y_val,
                'first_minutes': first_minutes_val,
                'prediction_type': pred_type,
                'loss_type': loss_type_val,
                'scale_factor': scale_factor,
            }
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            tmp_path = tmp_file.name
            tmp_file.close()
            try:
                model.save(tmp_path, include_optimizer=False, save_format='h5')
            except:
                tmp_path = None
            final_model_path = tmp_path

            if is_last:
                # Calculer DA finale
                if y_pred is not None and testY is not None:
                    baseline = 1.0 if pred_type == 'price' else 0.0
                    true_dir = np.sign(testY - baseline)
                    pred_dir = np.sign(y_pred - baseline)
                    da_main = float((true_dir == pred_dir).mean())
                
                predictions_flat_main = preds_flat
                predictions_train_flat_main = preds_train_flat
                
                final_predictions_data = {
                    'y_pred_test': y_pred.tolist() if y_pred is not None else [],
                    'y_true_test': testY.tolist() if testY is not None else [],
                    'predictions_flat': predictions_flat_main,
                    'predictions_train_flat': predictions_train_flat_main,
                    'look_back': look_back_val,
                    'stride': stride_val,
                    'nb_y': nb_y_val,
                    'first_minutes': first_minutes_val,
                    'prediction_type': pred_type,
                    'directional_accuracy': da_main,
                    'num_epochs': num_epochs,
                    'loss_type': loss_type_val,
                }
            else:
                extra_predictions.append({
                    'test': preds_flat,
                    'name': f'Win={look_back_val}',
                    'color': colors[i % len(colors)]
                })

        # Fin de boucle
        loss_label = {'mse': 'MSE', 'scaled_mse': 'Scaled MSE', 'mae': 'MAE'}.get(loss_type_val, 'MSE')
        title = f'📊 Prédictions ({loss_label})'
        if da_main: title += f' — DA={da_main*100:.1f}%'

        seg_fig = _build_segments_graph_from_store(
            store_json, window_sizes[-1], stride_val, first_minutes_val, 
            predictions=predictions_flat_main, 
            nb_y=nb_y_val, 
            predictions_train=predictions_train_flat_main, 
            prediction_type=pred_type,
            extra_predictions=extra_predictions
        )
        seg_fig.update_layout(title=title)
        
        logging.info("[Training] Tout terminé.")
        return seg_fig, final_predictions_data, False, True, final_model_path
        
    except Exception as e:
        logging.error(f"[Training] Erreur: {e}")
        import traceback
        traceback.print_exc()
        empty_seg_fig = go.Figure()
        empty_seg_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title=f'❌ Erreur: {e}', height=420, uirevision='play_segments')
        return empty_seg_fig, None, True, False, None


# ============================================================================
# CALLBACK BACKTEST SÉPARÉ
# ============================================================================

@app.callback(
    [
        Output('play_equity_graph', 'figure'),
        Output('play_trades_table', 'children'),
        Output('play_summary', 'children'),
    ],
    [Input('play_run_backtest', 'n_clicks')],
    [
        State('play_df_store', 'data'),
        State('play_predictions_store', 'data'),
        State('play_initial_cash', 'value'),
        State('play_trade_amount', 'value'),
        State('play_k_trades', 'value'),
        State('play_spread_pct', 'value'),
        State('play_strategy', 'value'),
    ],
    prevent_initial_call=True,
)
def run_backtest(n_clicks, store_json, predictions_data, initial_cash, per_trade, k_trades, spread_pct, strategy):
    """
    Exécute le backtest basé sur les prédictions stockées.
    
    Stratégies:
    - 'long': Acheter si hausse prédite (BUY → SELL)
    - 'short': Vendre si baisse prédite (SELL → BUY, short selling)
    - 'both': Long si hausse, Short si baisse
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' }, title='Équité — en attente', height=400, uirevision='play_equity')
    
    if not n_clicks or not store_json or not predictions_data:
        return empty_fig, html.Div('Entraînez d\'abord un modèle'), html.Div('')
    
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
        initial_cash_val = float(initial_cash or DEFAULT_INITIAL_CASH)
        per_trade_val = float(per_trade or DEFAULT_TRADE_AMOUNT)
        k_trades_val = int(k_trades or DEFAULT_K_TRADES)
        spread_pct_val = float(spread_pct or DEFAULT_SPREAD_PCT)
        strategy_val = strategy or 'long'
        
        # Récupérer les prédictions
        y_pred_test = predictions_data.get('y_pred_test')
        look_back = predictions_data.get('look_back', DEFAULT_LOOK_BACK)
        nb_y = predictions_data.get('nb_y', DEFAULT_NB_Y)
        pred_type = predictions_data.get('prediction_type', 'return')
        
        if not y_pred_test:
            return empty_fig, html.Div('Pas de prédictions disponibles'), html.Div('')
        
        logging.info(f"[Backtest] Démarrage: {len(y_pred_test)} prédictions, K={k_trades_val}/jour, stratégie={strategy_val}")
        
        # Baseline pour déterminer hausse/baisse
        baseline = 1.0 if pred_type == 'price' else 0.0
        
        equity_curve_times = []
        equity_curve_values = []
        trades = []
        cash = initial_cash_val
        
        idx = df.index
        days = idx.normalize().unique()
        split_idx = int(len(days) * 0.8)
        test_days = days[split_idx:]
        
        logging.info(f"[Backtest] Jours de test: {len(test_days)}, Prédictions: {len(y_pred_test)}")
        
        pred_idx = 0
        for day in test_days:
            if pred_idx >= len(y_pred_test):
                break
                
            mask = (idx.normalize() == day)
            day_df = df.loc[mask]
            if len(day_df) <= look_back + nb_y:
                continue
            
            # Prédictions pour ce jour
            y_pred_day = y_pred_test[pred_idx]
            pred_idx += 1
            y_pred_array = np.array(y_pred_day)
            
            # Calculer les offsets de sortie (répartis sur la journée)
            remainder = len(day_df) - look_back
            stride_y = max(1, remainder // (nb_y + 1))
            offsets = [(j + 1) * stride_y for j in range(nb_y)]
            
            # Sélectionner les trades selon la stratégie
            candidates = []
            for j in range(len(y_pred_array)):
                pred_value = float(y_pred_array[j])
                is_up = pred_value > baseline
                is_down = pred_value < baseline
                
                # Calculer l'amplitude de la prédiction (distance au baseline)
                amplitude = abs(pred_value - baseline)
                
                if strategy_val == 'long' and is_up:
                    candidates.append((j, 'LONG', amplitude, pred_value))
                elif strategy_val == 'short' and is_down:
                    candidates.append((j, 'SHORT', amplitude, pred_value))
                elif strategy_val == 'both':
                    if is_up:
                        candidates.append((j, 'LONG', amplitude, pred_value))
                    elif is_down:
                        candidates.append((j, 'SHORT', amplitude, pred_value))
            
            # Trier par amplitude décroissante (les plus forts signaux en premier)
            candidates.sort(key=lambda x: -x[2])
            
            # Sélectionner K trades NON CHEVAUCHANTS
            day_trades = []
            occupied_ranges = []  # Liste de (entry_idx, exit_idx) pour éviter les chevauchements
            
            for j, direction, amplitude, pred_value in candidates:
                if len(day_trades) >= k_trades_val:
                    break
                
                # Calculer les indices d'entrée et de sortie
                entry_idx = look_back
                off = int(offsets[j]) if j < len(offsets) else stride_y * (j + 1)
                exit_idx = min(entry_idx + off, len(day_df) - 1)
                
                # Vérifier qu'il n'y a pas de chevauchement
                overlaps = False
                for (occ_entry, occ_exit) in occupied_ranges:
                    # Chevauchement si les intervalles se croisent
                    if not (exit_idx <= occ_entry or entry_idx >= occ_exit):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                # Ajouter ce trade
                occupied_ranges.append((entry_idx, exit_idx))
                
                entry_time = day_df.index[entry_idx]
                exit_time = day_df.index[exit_idx]
                mid_entry_price = float(day_df.iloc[entry_idx]['openPrice'])
                mid_exit_price = float(day_df.iloc[exit_idx]['openPrice'])
                
                # Spread
                half_spread = spread_pct_val / 100.0 / 2.0
                
                # Calculer P&L selon la direction
                qty = int(per_trade_val // max(1e-9, mid_entry_price))
                if qty <= 0:
                    continue
                
                if direction == 'LONG':
                    # LONG: Acheter au ask, vendre au bid
                    entry_price = mid_entry_price * (1 + half_spread)
                    exit_price = mid_exit_price * (1 - half_spread)
                    pnl = float((exit_price - entry_price) * qty)
                else:
                    # SHORT: Vendre au bid, racheter au ask
                    entry_price = mid_entry_price * (1 - half_spread)  # Prix de vente
                    exit_price = mid_exit_price * (1 + half_spread)    # Prix de rachat
                    pnl = float((entry_price - exit_price) * qty)      # Gain si le prix baisse
                
                day_trades.append({
                    'entry_time': str(entry_time),
                    'exit_time': str(exit_time),
                    'direction': direction,
                    'qty': qty,
                    'entry_price': round(entry_price, 4),
                    'exit_price': round(exit_price, 4),
                    'predicted': round(pred_value, 6),
                    'pnl': round(pnl, 2)
                })
                
                cash += pnl
                equity_curve_times.append(exit_time)
                equity_curve_values.append(cash)
            
            trades.extend(day_trades)
            if day_trades:
                logging.debug(f"[Backtest] {day.date()}: {len(day_trades)} trades ({strategy_val})")
        
        logging.info(f"[Backtest] Terminé: {len(trades)} trades, Cash final: {cash:.2f}€")
        
        # Figures et tableaux
        if equity_curve_times:
            eq_fig = build_equity_figure('model', 'SYNTH', None, None, None, None, None, equity_curve_times, equity_curve_values)
        else:
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[initial_cash_val, initial_cash_val], mode='lines', name='Cash initial'))
        
        pct_return = ((cash / initial_cash_val) - 1) * 100
        eq_fig.update_layout(
            template='plotly_dark', paper_bgcolor='#000', plot_bgcolor='#000', font={ 'color': '#FFF' },
            title=f'💰 Équité: {cash:,.2f}€ ({pct_return:+.2f}%) — {len(trades)} trades',
            height=400, uirevision='play_equity'
        )
        
        # Tableau des trades amélioré
        trades_table = _build_trades_table_v2(trades)
        
        # Résumé
        total_pnl = cash - initial_cash_val
        num_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        
        strategy_labels = {'long': 'LONG (hausse)', 'short': 'SHORT (baisse)', 'both': 'LONG & SHORT'}
        long_count = len([t for t in trades if t.get('direction') == 'LONG'])
        short_count = len([t for t in trades if t.get('direction') == 'SHORT'])
        
        summary_items = [
            f"💰 Capital final: {cash:,.2f}€",
            f"📊 P&L total: {total_pnl:+,.2f}€ ({pct_return:+.2f}%)",
            f"📈 Trades: {num_trades} — Win rate: {win_rate:.1f}%",
            f"🎯 Stratégie: {strategy_labels.get(strategy_val, strategy_val)} (📈{long_count} / 📉{short_count})",
            f"💵 Spread: {spread_pct_val:.2f}% — K={k_trades_val}/jour",
        ]
        summary = html.Ul([html.Li(it) for it in summary_items], style={ 'color': '#FFFFFF' })
        
        return eq_fig, trades_table, summary
        
    except Exception as e:
        logging.error(f"[Backtest] Erreur: {e}")
        import traceback
        traceback.print_exc()
        empty_fig.update_layout(title=f'❌ Erreur backtest: {e}')
        return empty_fig, html.Div(f'Erreur: {e}'), html.Div('')


@app.callback(
    [
        Output('play_segments_graph', 'figure', allow_duplicate=True),
        Output('play_gen_summary', 'children'),
    ],
    [
        Input('play_test_generalization', 'n_clicks'),
    ],
    [
        State('play_df_store', 'data'),
        State('play_model_path', 'data'),
        State('play_model_ready', 'data'),
    ],
    prevent_initial_call=True,
)
def test_generalization_on_current_curve(n_clicks, store_json, model_path, model_ready):
    """
    Applique le dernier modèle entraîné sur la courbe synthétique actuellement affichée
    pour vérifier sa capacité de généralisation (mêmes X premières minutes, reste de la journée différent).
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={ 'color': '#FFFFFF' },
        title='Série synthétique — en attente de généralisation',
        height=420,
        uirevision='play_segments'
    )

    if not n_clicks:
        return empty_fig, html.Div("Cliquez sur le bouton pour lancer le test de généralisation.", style={ 'color': '#CCCCCC' })

    global play_last_model, play_last_model_meta, play_last_model_path

    if not model_ready:
        return empty_fig, html.Div("Aucun modèle en mémoire. Entraînez d'abord un modèle dans le panneau de droite.", style={ 'color': '#F59E0B' })

    # Charger le modèle depuis le disque si nécessaire
    if play_last_model is None:
        if not model_path:
            return empty_fig, html.Div("Aucun modèle en mémoire. Entraînez d'abord un modèle dans le panneau de droite.", style={ 'color': '#F59E0B' })
        try:
            custom_objects = get_custom_objects() if TRANSFORMER_AVAILABLE else None
            play_last_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            play_last_model_path = model_path
            logging.info(f"[Generalization] Modèle rechargé depuis {model_path}")
        except Exception as load_err:
            logging.error(f"[Generalization] Impossible de recharger le modèle: {load_err}")
            return empty_fig, html.Div("Impossible de recharger le modèle. Ré-entraînez-le.", style={ 'color': '#EF4444' })

    if not store_json:
        return empty_fig, html.Div("Aucune courbe synthétique disponible. Générez d'abord une courbe (par exemple sinusoïdale).", style={ 'color': '#F59E0B' })

    try:
        look_back = int(play_last_model_meta.get('look_back', DEFAULT_LOOK_BACK))
        stride = int(play_last_model_meta.get('stride', DEFAULT_STRIDE))
        nb_y = int(play_last_model_meta.get('nb_y', DEFAULT_NB_Y))
        first_minutes = int(play_last_model_meta.get('first_minutes', DEFAULT_FIRST_MINUTES))
        prediction_type = play_last_model_meta.get('prediction_type', 'return')
        loss_type = play_last_model_meta.get('loss_type', 'mse')
        scale_factor = float(play_last_model_meta.get('scale_factor', 1.0))

        X, Y, df, obs_window, sample_days = _prepare_xy_for_inference(
            store_json,
            look_back,
            stride,
            nb_y,
            first_minutes,
            prediction_type,
        )

        if X is None or X.shape[0] == 0:
            return empty_fig, html.Div("Données insuffisantes pour construire les fenêtres d'entrée sur cette courbe.", style={ 'color': '#F59E0B' })

        # Prédictions du modèle sur cette nouvelle courbe
        y_pred = play_last_model.predict(X, verbose=0)

        if loss_type == 'scaled_mse' and scale_factor != 0.0:
            y_pred = y_pred / scale_factor
            if Y is not None:
                Y = Y / scale_factor

        # Graphe de base (train/test/pred originales) pour conserver les couleurs existantes
        fig_base = _build_segments_graph_from_store(
            store_json,
            look_back,
            stride,
            first_minutes,
            None,
            nb_y,
            None,
            prediction_type,
        )

        # Traces de généralisation (points réels/pred)
        fig_gen, mae, rmse, n_points = _build_generalization_figure(
            df,
            sample_days,
            obs_window,
            y_pred,
            nb_y,
            prediction_type,
        )

        if n_points == 0:
            return fig_base, html.Div("Impossible de reconstruire des points futurs sur cette courbe (fenêtre trop courte ?).", style={ 'color': '#F59E0B' })

        # Injecter les traces (sauf la série de fond) dans le graphe principal
        for tr in fig_gen.data:
            if getattr(tr, 'name', '') == 'Série (nouvelle)':
                continue
            fig_base.add_trace(tr)

        # Légende lisible
        fig_base.update_layout(
            legend=dict(
                font=dict(color='#FFFFFF'),
                bgcolor='rgba(0,0,0,0.35)',
                bordercolor='#444'
            ),
            title="Série synthétique & Segments (avec généralisation)"
        )

        summary = html.Div([
            html.Div(f"Nombre de points évalués : {n_points}"),
            html.Div(f"MAE (erreur absolue moyenne) : {mae:.4f}"),
            html.Div(f"RMSE (erreur quadratique moyenne) : {rmse:.4f}"),
        ], style={ 'color': '#FFFFFF' })

        return fig_base, summary
    except Exception as e:
        logging.error(f"[Generalization] Erreur: {e}")
        return empty_fig, html.Div(f"Erreur durant le test de généralisation: {e}", style={ 'color': '#EF4444' })


def _build_trades_table_v2(trades):
    """Construit un tableau de trades avec direction, heures d'entrée/sortie."""
    if not trades:
        return html.Div('Aucun trade effectué', style={ 'color': '#888', 'padding': '8px' })
    
    rows = []
    for i, t in enumerate(trades[-30:], 1):  # 30 derniers trades
        pnl = t.get('pnl', 0)
        pnl_color = '#4CAF50' if pnl > 0 else '#f44336' if pnl < 0 else '#888'
        pred_val = t.get('predicted', 0)
        direction = t.get('direction', 'LONG')
        
        # Couleur selon direction
        dir_color = '#4CAF50' if direction == 'LONG' else '#f44336'
        dir_icon = '📈' if direction == 'LONG' else '📉'
        
        # Extraire date et heures
        entry_time = t.get('entry_time', '-')
        exit_time = t.get('exit_time', '-')
        
        # Format: "2024-01-15 10:30:00" -> date + heures
        entry_dt = entry_time[:10] if len(entry_time) >= 10 else '-'
        entry_hr = entry_time[11:16] if len(entry_time) >= 16 else '-'
        exit_hr = exit_time[11:16] if len(exit_time) >= 16 else '-'
        
        rows.append(html.Tr([
            html.Td(str(i), style={ 'padding': '4px 6px', 'textAlign': 'center' }),
            html.Td(f"{dir_icon}", style={ 'padding': '4px 6px', 'textAlign': 'center', 'color': dir_color, 'fontSize': '14px' }),
            html.Td(entry_dt, style={ 'padding': '4px 6px' }),
            html.Td(entry_hr, style={ 'padding': '4px 6px', 'textAlign': 'center' }),
            html.Td(exit_hr, style={ 'padding': '4px 6px', 'textAlign': 'center' }),
            html.Td(f"{t.get('qty', 0)}", style={ 'padding': '4px 6px', 'textAlign': 'right' }),
            html.Td(f"{t.get('entry_price', 0):.2f}", style={ 'padding': '4px 6px', 'textAlign': 'right' }),
            html.Td(f"{t.get('exit_price', 0):.2f}", style={ 'padding': '4px 6px', 'textAlign': 'right' }),
            html.Td(f"{pnl:+.2f}€", style={ 'padding': '4px 6px', 'textAlign': 'right', 'color': pnl_color, 'fontWeight': 'bold' }),
        ]))
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th('#', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center' }),
            html.Th('Dir', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center' }),
            html.Th('Date', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444' }),
            html.Th('Entrée', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center' }),
            html.Th('Sortie', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center' }),
            html.Th('Qté', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right' }),
            html.Th('P.Entrée', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right' }),
            html.Th('P.Sortie', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right' }),
            html.Th('P&L', style={ 'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right' }),
        ])),
        html.Tbody(rows)
    ], style={ 'width': '100%', 'color': '#FFF', 'fontSize': '11px', 'borderCollapse': 'collapse' })
