#dashboard.py
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
from web.components.navigation import create_navigation, create_page_help

# Styles communs pour les cartes
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'transition': 'all 0.25s ease'
}

METRIC_CARD_STYLE = {
    **CARD_STYLE,
    'flex': '1',
    'minWidth': '200px',
    'position': 'relative',
    'overflow': 'hidden'
}

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
                # Si on n'a pas assez d'actions, ignorer cette vente
                print(f"Warning: Attempted to sell {row['quantity']} shares of {row['symbol']} but only had {current_holdings[row['symbol']]['quantity']}")
                continue
            
    # Calculer les gains/pertes pour différentes périodes
    now = datetime.now()
    periods = {
        'day': ('24h', now - timedelta(days=1)),
        'week': ('7j', now - timedelta(days=7)),
        'month': ('30j', now - timedelta(days=30)),
        'all_time': ('Total', df['date'].min())
    }
    
    results = {}
    for period_key, (period_label, start_date) in periods.items():
        period_df = df[df['date'] >= start_date]
        gains = period_df[period_df['type'] == 'SELL']['total'].sum()
        losses = period_df[period_df['type'] == 'BUY']['total'].sum()
        net = gains - losses
        results[period_key] = {
            'label': period_label,
            'gains': gains,
            'losses': losses,
            'net': net
        }
    
    return current_holdings, results

def create_pnl_graph(df):
    """Créer un graphique d'évolution des gains/pertes cumulés avec un style moderne."""
    df = df.copy()
    df['cumulative_pnl'] = df.apply(
        lambda x: x['total'] if x['type'] == 'SELL' else -x['total'],
        axis=1
    ).cumsum()
    
    # Création du graphique avec gradient
    fig = go.Figure()
    
    # Zone de remplissage sous la courbe
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_pnl'],
        fill='tozeroy',
        mode='lines',
        name='P&L Cumulé',
        line=dict(color='#8b5cf6', width=2),
        fillcolor='rgba(139, 92, 246, 0.1)'
    ))
    
    # Ligne principale
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cumulative_pnl'],
        mode='lines+markers',
        name='P&L',
        line=dict(color='#a855f7', width=3),
        marker=dict(size=6, color='#a855f7')
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text='<b>Évolution des Gains/Pertes</b>',
            font=dict(size=18, color='#f8fafc'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        yaxis=dict(
            title=dict(text='P&L Cumulé ($)', font=dict(color='#94a3b8')),
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            tickfont=dict(color='#94a3b8')
        ),
        plot_bgcolor='#0a0a0f',
        paper_bgcolor='#0a0a0f',
        font=dict(family='Outfit, sans-serif', color='#f8fafc'),
        margin=dict(l=60, r=30, t=60, b=40),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_metric_card(title, metrics, period_key):
    """Créer une carte de métrique avec un design moderne."""
    net_color = '#10b981' if metrics['net'] >= 0 else '#ef4444'
    net_icon = '↑' if metrics['net'] >= 0 else '↓'
    
    # Gradient de fond en fonction du signe
    gradient_style = {
        'position': 'absolute',
        'top': 0,
        'left': 0,
        'right': 0,
        'height': '4px',
        'background': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
        'borderRadius': '16px 16px 0 0'
    }
    
    return html.Div([
        # Barre de gradient en haut
        html.Div(style=gradient_style),
        
        # Titre de la période
        html.Div([
            html.Span(metrics['label'], style={
                'fontSize': '0.875rem',
                'fontWeight': '500',
                'color': '#a78bfa',
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em'
            })
        ], style={'marginBottom': '16px'}),
        
        # Valeur nette principale
        html.Div([
            html.Span(net_icon, style={
                'fontSize': '1.5rem',
                'marginRight': '8px',
                'color': net_color
            }),
            html.Span(f"${abs(metrics['net']):,.2f}", style={
                'fontSize': '1.75rem',
                'fontWeight': '700',
                'color': net_color
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '16px'
        }),
        
        # Détails gains/pertes
        html.Div([
            html.Div([
                html.Span('Gains', style={
                    'fontSize': '0.75rem',
                    'color': '#64748b',
                    'display': 'block'
                }),
                html.Span(f"+${metrics['gains']:,.2f}", style={
                    'fontSize': '0.9375rem',
                    'fontWeight': '600',
                    'color': '#10b981'
                })
            ], style={'marginRight': '24px'}),
            html.Div([
                html.Span('Pertes', style={
                    'fontSize': '0.75rem',
                    'color': '#64748b',
                    'display': 'block'
                }),
                html.Span(f"-${metrics['losses']:,.2f}", style={
                    'fontSize': '0.9375rem',
                    'fontWeight': '600',
                    'color': '#ef4444'
                })
            ])
        ], style={
            'display': 'flex',
            'alignItems': 'flex-start'
        })
    ], style=METRIC_CARD_STYLE)

# Charger les données
df = load_transaction_data()
current_holdings, period_results = calculate_portfolio_metrics(df)
total_portfolio_value = sum(holding['total_cost'] for holding in current_holdings.values())

help_text = """
### Dashboard (Tableau de Bord)

Cette page est votre centre de contrôle pour suivre la santé financière de votre portefeuille en un coup d'œil.

#### 1. Valeur Totale
Le gros chiffre en haut indique la valeur actuelle de toutes vos positions ouvertes, basées sur leur coût d'achat initial. C'est votre engagement total sur le marché.

#### 2. Indicateurs de Performance (P&L)
Ces blocs colorés vous montrent combien vous avez gagné ou perdu sur différentes périodes :
*   **24h** : Performance des dernières 24 heures.
*   **7j** : Performance sur les 7 derniers jours.
*   **30j** : Performance sur les 30 derniers jours.
*   **Total** : Performance globale depuis le tout début.

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
    
    # En-tête de page
    html.Div([
        html.H3('Dashboard', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Vue d\'ensemble de votre portefeuille', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0'
        })
    ], style={'marginBottom': '32px'}),
    
    # Carte principale - Valeur du portefeuille
    html.Div([
        html.Div([
            html.Div([
                html.Span('Valeur Totale du Portefeuille', style={
                    'fontSize': '0.875rem',
                    'fontWeight': '500',
                    'color': '#94a3b8',
                    'textTransform': 'uppercase',
                    'letterSpacing': '0.1em',
                    'display': 'block',
                    'marginBottom': '12px'
                }),
                html.Div([
                    html.Span('$', style={
                        'fontSize': '2rem',
                        'fontWeight': '300',
                        'color': '#a78bfa',
                        'marginRight': '4px'
                    }),
                    html.Span(f"{total_portfolio_value:,.2f}", style={
                        'fontSize': '3.5rem',
                        'fontWeight': '700',
                        'background': 'linear-gradient(135deg, #f8fafc 0%, #a78bfa 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'backgroundClip': 'text'
                    })
                ], style={
                    'display': 'flex',
                    'alignItems': 'baseline'
                }),
                html.Span(f'{len(current_holdings)} positions actives', style={
                    'fontSize': '0.875rem',
                    'color': '#64748b',
                    'marginTop': '8px',
                    'display': 'block'
                })
            ])
        ], style={
            **CARD_STYLE,
            'textAlign': 'center',
            'background': 'linear-gradient(145deg, #1a1a24 0%, #12121a 100%)',
            'border': '1px solid rgba(139, 92, 246, 0.2)',
            'boxShadow': '0 0 30px rgba(99, 102, 241, 0.1)'
        })
    ], style={'marginBottom': '24px'}),
    
    # Cartes de métriques P&L
    html.Div([
        create_metric_card('day', period_results['day'], 'day'),
        create_metric_card('week', period_results['week'], 'week'),
        create_metric_card('month', period_results['month'], 'month'),
        create_metric_card('all_time', period_results['all_time'], 'all_time'),
    ], style={
        'display': 'flex',
        'gap': '16px',
        'flexWrap': 'wrap',
        'marginBottom': '24px'
    }),
    
    # Graphique P&L
    html.Div([
        dcc.Graph(
            figure=create_pnl_graph(df),
            style={'height': '50vh'},
            config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False
            }
        )
    ], style={
        **CARD_STYLE,
        'marginBottom': '24px'
    }),

    # Spacer pour la navigation
    html.Div(style={'height': '100px'}),

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
