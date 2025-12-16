# -*- coding: utf-8 -*-
"""
Textes d'aide centralisés pour toutes les pages.
Permet de réutiliser et maintenir les contenus d'aide facilement.
"""

# ==============================================================================
# SECTIONS D'AIDE COMMUNES (RÉUTILISABLES)
# ==============================================================================

HELP_MODELS_IA = """
#### Les 3 Types de Modèles IA

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
"""

HELP_DATA_PARAMS = """
#### Paramètres de Données

- `look_back` : Combien de minutes passées le modèle voit (60 = 1h)
- `stride` : Échantillonnage (stride=5 → 1 point toutes les 5 min)
- `nb_y` : Combien de points futurs prédire
- `Premières minutes` : Période d'observation avant de trader
"""

HELP_LOSS_TYPES = """
#### Types de Loss (Fonction de Perte)

Le choix de la loss influence l'entraînement et la lisibilité des métriques :

| Type | Description | Avantages | Inconvénients |
|------|-------------|-----------|---------------|
| **MSE** | Mean Squared Error (défaut) | Standard, stable | Valeurs très petites (10⁻⁶) si variations faibles |
| **Scaled MSE** | MSE × 100 | Loss lisible (~0.01-1.0), même comportement que MSE | Prédictions à rescaler mentalement |
| **MAE** | Mean Absolute Error | Robuste aux outliers, même unité que les targets | Moins pénalisant pour les grosses erreurs |

**Recommandation :** Utilisez **Scaled MSE** si le loss standard est illisible (< 0.0001).
"""

HELP_TRADING_STRATEGIES = """
#### Stratégies de Trading (Backtest)

| Stratégie | Description | Quand l'utiliser |
|-----------|-------------|------------------|
| **📈 LONG** | Acheter si hausse prédite → Vendre plus tard | Marché haussier ou pattern de hausse |
| **📉 SHORT** | Vendre si baisse prédite → Racheter moins cher | Marché baissier ou pattern de baisse |
| **📊 LONG & SHORT** | Les deux selon la prédiction | Maximum d'opportunités |

- Les trades ne se chevauchent **jamais** sur une même journée
- Chaque jour, jusqu'à **K trades** sont exécutés parmi les prédictions les plus fortes
- Le **spread** est appliqué sur chaque trade (coût réaliste)
"""


# ==============================================================================
# TEXTES D'AIDE PAR PAGE
# ==============================================================================

def get_playground_help():
    """Texte d'aide pour la page Playground."""
    return f"""
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
*   **📊 Plateau (N niveaux)** : N paliers fixes qui se répètent chaque jour :
    - **Matin** (1er tiers) : Prix de base
    - **Midi** (2ème tiers) : Prix + amplitude
    - **Après-midi** (3ème tiers) : Prix - amplitude/2
    - *Idéal pour tester si le modèle détecte les patterns répétitifs !*

---

{HELP_MODELS_IA}

---

{HELP_DATA_PARAMS}

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

{HELP_LOSS_TYPES}

---

{HELP_TRADING_STRATEGIES}

---

#### 6. Résultats

- **Série synthétique** : La courbe générée avec les prédictions
- **Équité** : Évolution du portefeuille selon la stratégie
- **Tableau des trades** : Détail avec direction (📈/📉), heures entrée/sortie, P&L
- **Historique** : Loss (échelle log) et Directional Accuracy pendant l'entraînement
"""


def get_prediction_help():
    """Texte d'aide pour la page Prediction."""
    return f"""
### Prédiction (Deep Learning)

Cette page est le cœur du système d'intelligence artificielle. Elle permet de créer, entraîner et évaluer des modèles de prédiction sur des données réelles.

---

{HELP_MODELS_IA}

---

{HELP_DATA_PARAMS}

---

{HELP_LOSS_TYPES}

---

#### Workflow de prédiction

1. **Sélectionner les actions** à analyser
2. **Configurer les paramètres** du modèle (architecture, hyperparamètres)
3. **Lancer l'entraînement** et suivre la progression
4. **Évaluer les résultats** (métriques, graphiques)
5. **Sauvegarder le modèle** pour utilisation ultérieure
"""


def get_simulation_help():
    """Texte d'aide pour la page Simulation."""
    return f"""
### Simulation (Backtesting)

Cette page est dédiée au test de stratégies de trading sur des données historiques pour évaluer leur rentabilité potentielle avant de les utiliser en réel.

---

{HELP_TRADING_STRATEGIES}

---

#### Métriques de performance

- **Rendement total** : Gain/perte en pourcentage sur la période
- **Win rate** : Pourcentage de trades gagnants
- **Sharpe ratio** : Rendement ajusté au risque
- **Max drawdown** : Perte maximale depuis un pic
"""


def get_analyse_help():
    """Texte d'aide pour la page Analyse."""
    return """
### Analyse de Données

Cette page est votre tableau de bord statistique pour comprendre les dynamiques du marché et les relations entre les actions.

---

#### Fonctionnalités

- **Statistiques descriptives** : Min, max, moyenne, écart-type
- **Corrélations** : Matrice de corrélation entre actions
- **Distribution** : Histogrammes des rendements
- **Tendances** : Moyennes mobiles et indicateurs
"""


def get_dashboard_help():
    """Texte d'aide pour la page Dashboard."""
    return """
### Dashboard (Tableau de Bord)

Cette page est votre centre de contrôle pour suivre la santé financière de votre portefeuille en un coup d'œil.

---

#### Indicateurs clés

- **Valeur du portefeuille** : Évolution dans le temps
- **Performance journalière** : Gains/pertes du jour
- **Répartition** : Allocation par secteur/action
- **Alertes** : Seuils de prix atteints
"""


def get_visualisation_help():
    """Texte d'aide pour la page Visualisation."""
    return """
### Visualisation des Données

Cette page vous permet d'explorer visuellement les données historiques de vos actions.

---

#### Graphiques disponibles

- **Chandeliers** : Prix OHLC (Open, High, Low, Close)
- **Volumes** : Histogramme des échanges
- **Indicateurs techniques** : RSI, MACD, Bollinger
- **Comparaison** : Superposition de plusieurs actions
"""

