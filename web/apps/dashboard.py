#dashboard.py
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from web.components.navigation import create_navigation, create_page_help  # Importer le composant d'aide

def load_transaction_data():
    try:
        return pd.read_csv('mock_transactions.csv', parse_dates=['date'])
    except FileNotFoundError:
        from web.data.mock_transactions import generate_mock_transactions
        data = generate_mock_transactions()
        data.to_csv('mock_transactions.csv', index=False)
        return data

def calculate_portfolio_metrics(df):
    # Calculer la valeur totale du portefeuille
    portfolio_value = 0
    current_holdings = {}
    
    # Initialiser les holdings pour tous les symboles uniques
    for symbol in df['symbol'].unique():
        current_holdings[symbol] = {'quantity': 0, 'total_cost': 0}
    
    # Trier le DataFrame par date pour traiter les transactions chronologiquement
    df = df.sort_values('date')
    
    for _, row in df.iterrows():
        if row['type'] == 'BUY':
            current_holdings[row['symbol']]['quantity'] += row['quantity']
            current_holdings[row['symbol']]['total_cost'] += row['total']
        else:  # SELL
            # Vérifier si nous avons assez d'actions à vendre
            if current_holdings[row['symbol']]['quantity'] >= row['quantity']:
                current_holdings[row['symbol']]['quantity'] -= row['quantity']
                # Ajuster le coût total proportionnellement
                cost_per_share = current_holdings[row['symbol']]['total_cost'] / (current_holdings[row['symbol']]['quantity'] + row['quantity'])
                current_holdings[row['symbol']]['total_cost'] -= (cost_per_share * row['quantity'])
            else:
                # Si on n'a pas assez d'actions, ignorer cette vente ou la traiter comme une vente à découvert
                print(f"Warning: Attempted to sell {row['quantity']} shares of {row['symbol']} but only had {current_holdings[row['symbol']]['quantity']}")
                continue
            
    # Calculer les gains/pertes pour différentes périodes
    now = datetime.now()
    periods = {
        'all_time': df['date'].min(),
        'month': now - timedelta(days=30),
        'week': now - timedelta(days=7),
        'day': now - timedelta(days=1)
    }
    
    results = {}
    for period_name, start_date in periods.items():
        period_df = df[df['date'] >= start_date]
        gains = period_df[period_df['type'] == 'SELL']['total'].sum()
        losses = period_df[period_df['type'] == 'BUY']['total'].sum()
        net = gains - losses
        results[period_name] = {
            'gains': gains,
            'losses': losses,
            'net': net
        }
    
    return current_holdings, results

def create_pnl_graph(df):
    # Créer un graphique d'évolution des gains/pertes cumulés
    df['cumulative_pnl'] = df.apply(
        lambda x: x['total'] if x['type'] == 'SELL' else -x['total'],
        axis=1
    ).cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_pnl'],
        mode='lines',
        name='P&L Cumulé',
        line=dict(color='white')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title='Évolution des Gains/Pertes',
        xaxis_title='Date',
        yaxis_title='P&L Cumulé ($)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    
    return fig

# Charger les données
df = load_transaction_data()
current_holdings, period_results = calculate_portfolio_metrics(df)

help_text = """
### Dashboard (Tableau de Bord)

Cette page est votre centre de contrôle pour suivre la santé financière de votre portefeuille en un coup d'œil.

#### 1. Valeur Totale
Le gros chiffre en haut indique la valeur actuelle de toutes vos positions ouvertes, basées sur leur coût d'achat initial. C'est votre engagement total sur le marché.

#### 2. Indicateurs de Performance (P&L)
Ces blocs colorés vous montrent combien vous avez gagné ou perdu sur différentes périodes :
*   **Day** : Performance des dernières 24h.
*   **Week** : Performance sur les 7 derniers jours.
*   **Month** : Performance sur les 30 derniers jours.
*   **All Time** : Performance globale depuis le tout début.

*Pour chaque période :*
*   **Gains (Vert)** : Argent encaissé grâce aux ventes.
*   **Pertes (Rouge)** : Argent sorti pour les achats.
*   **Net** : Le résultat final (Gains - Pertes). Si c'est vert, vous êtes rentable sur la période !

#### 3. Évolution des Gains/Pertes
Ce graphique trace votre courbe de profitabilité cumulée.
*   Une courbe qui monte régulièrement vers la droite est signe d'une stratégie saine.
*   Les pics vers le bas indiquent des périodes d'investissement (achat).
*   Les pics vers le haut indiquent des prises de profits (vente).
"""

layout = html.Div([
    create_page_help("Aide Dashboard", help_text),
    html.H3('Dashboard', style={'color': 'white', 'textAlign': 'center'}),
    
    # Portfolio Value Section
    html.Div([
        html.H4('Valeur Totale du Portefeuille', style={'color': 'white'}),
        html.H2(
            f"${sum(holding['total_cost'] for holding in current_holdings.values()):,.2f}",
            style={'color': 'white'}
        )
    ], style={'textAlign': 'center', 'margin': '20px'}),
    
    # P&L Metrics for Different Periods
    html.Div([
        html.Div([
            html.H5(period.title(), style={'color': 'white'}),
            html.Div([
                html.P(f"Gains: ${metrics['gains']:,.2f}", style={'color': '#4CAF50'}),
                html.P(f"Pertes: ${metrics['losses']:,.2f}", style={'color': '#f44336'}),
                html.P(
                    f"Net: ${metrics['net']:,.2f}",
                    style={'color': '#4CAF50' if metrics['net'] > 0 else '#f44336'}
                )
            ])
        ], style={
            'backgroundColor': '#1a1a1a',
            'padding': '15px',
            'borderRadius': '5px',
            'margin': '10px',
            'flex': '1'
        }) for period, metrics in period_results.items()
    ], style={
        'display': 'flex',
        'justifyContent': 'space-around',
        'flexWrap': 'wrap',
        'margin': '20px 0'
    }),
    
    # P&L Evolution Graph
    html.Div([
        dcc.Graph(
            figure=create_pnl_graph(df),
            style={'height': '50vh'}
        )
    ], style={'margin': '20px 0'}),

    # Spacer pour éviter que le contenu ne soit caché derrière la navigation
    html.Div(style={'height': '80px'}),

    # Navigation standardisée
    create_navigation()

], style={
    'backgroundColor': 'black',
    'minHeight': '100vh',
    'padding': '20px'
})
