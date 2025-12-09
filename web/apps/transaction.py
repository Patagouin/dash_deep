from dash import dcc, html
from dash.dependencies import Input, Output, State
from web.components.navigation import create_navigation, create_page_help
import pandas as pd
from Models.trading212 import buy_stock, sell_stock, get_account_info
from app import app, shM

# Styles communs
CARD_STYLE = {
    'backgroundColor': '#1a1a24',
    'padding': '24px',
    'borderRadius': '16px',
    'border': '1px solid rgba(148, 163, 184, 0.1)',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    'marginBottom': '24px'
}

BUTTON_BUY = {
    'background': 'linear-gradient(135deg, #10b981 0%, #34d399 100%)',
    'color': 'white',
    'padding': '14px 40px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '1rem',
    'fontWeight': '700',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 15px rgba(16, 185, 129, 0.3)',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em'
}

BUTTON_SELL = {
    'background': 'linear-gradient(135deg, #ef4444 0%, #f87171 100%)',
    'color': 'white',
    'padding': '14px 40px',
    'border': 'none',
    'borderRadius': '10px',
    'cursor': 'pointer',
    'fontSize': '1rem',
    'fontWeight': '700',
    'fontFamily': 'Outfit, sans-serif',
    'transition': 'all 0.25s ease',
    'boxShadow': '0 4px 15px rgba(239, 68, 68, 0.3)',
    'textTransform': 'uppercase',
    'letterSpacing': '0.05em'
}

help_text = """
### Transactions

Cette page permet d'exécuter des ordres d'achat et de vente sur votre broker.

#### Fonctionnalités
*   **Sélection d'action** : Choisissez l'action à trader dans la liste déroulante.
*   **Achat (Buy)** : Achète la quantité spécifiée de l'action sélectionnée.
*   **Vente (Sell)** : Vend la quantité spécifiée de l'action sélectionnée.

#### Important
*   Assurez-vous d'avoir configuré votre API key dans la page Configuration.
*   Les transactions sont exécutées en temps réel sur votre compte broker.
"""

layout = html.Div([
    create_page_help("Aide Transactions", help_text),
    
    # En-tête de page
    html.Div([
        html.H3('Transactions', style={
            'margin': '0',
            'textAlign': 'center'
        }),
        html.P('Exécution des ordres d\'achat et de vente', style={
            'textAlign': 'center',
            'color': '#94a3b8',
            'marginTop': '8px',
            'marginBottom': '0'
        })
    ], style={'marginBottom': '32px'}),

    # Carte de trading
    html.Div([
        html.Div([
            html.Span('💹', style={'fontSize': '1.5rem'}),
            html.Span('Passer un Ordre', style={
                'fontSize': '1.25rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '12px'
            })
        ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
        
        # Sélection d'action
        html.Div([
            html.Label('Sélectionner une action', style={
                'display': 'block',
                'fontSize': '0.875rem',
                'fontWeight': '500',
                'color': '#94a3b8',
                'marginBottom': '8px',
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em'
            }),
            dcc.Dropdown(
                id='transaction_dropdown',
                options=[
                    {'label': f'📈 {stock.symbol}', 'value': stock.symbol} 
                    for stock in sorted(shM.dfShares.itertuples(), key=lambda stock: stock.symbol)
                ],
                multi=False,
                placeholder='Rechercher une action...',
                style={'marginBottom': '24px'},
                persistence=True, 
                persistence_type='session'
            ),
        ]),
        
        # Boutons Buy/Sell
        html.Div([
            html.Button('🛒 BUY', id='buy_button', n_clicks=0, style=BUTTON_BUY),
            html.Button('💰 SELL', id='sell_button', n_clicks=0, style=BUTTON_SELL)
        ], style={
            'display': 'flex',
            'justifyContent': 'center',
            'gap': '24px',
            'marginTop': '8px'
        }),
    ], style={
        **CARD_STYLE,
        'maxWidth': '500px',
        'margin': '0 auto 32px',
        'textAlign': 'center'
    }),

    # Carte des transactions
    html.Div([
        html.Div([
            html.Span('📋', style={'fontSize': '1.25rem'}),
            html.Span('Historique des Transactions', style={
                'fontSize': '1.125rem',
                'fontWeight': '600',
                'color': '#a78bfa',
                'marginLeft': '10px'
            })
        ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
        
        html.Div(
            id='transaction_table',
            style={
                'backgroundColor': '#12121a',
                'borderRadius': '12px',
                'padding': '16px',
                'minHeight': '200px'
            }
        )
    ], style={
        **CARD_STYLE,
        'maxWidth': '900px',
        'margin': '0 auto 24px'
    }),

    # Spacer pour navigation
    html.Div(style={'height': '100px'}),

    # Navigation
    create_navigation()
], style={
    'backgroundColor': '#0a0a0f',
    'minHeight': '100vh',
    'padding': '24px'
})

# Callback pour gérer les achats et ventes
@app.callback(
    Output('transaction_table', 'children'),
    Input('buy_button', 'n_clicks'),
    Input('sell_button', 'n_clicks'),
    State('transaction_dropdown', 'value')
)
def handle_transaction(buy_clicks, sell_clicks, selected_stock):
    if not selected_stock:
        return html.Div([
            html.Div([
                html.Span('⚠️', style={'fontSize': '2rem', 'marginBottom': '12px', 'display': 'block'}),
                html.Span('Veuillez sélectionner une action', style={
                    'color': '#f59e0b',
                    'fontSize': '1rem',
                    'fontWeight': '500'
                })
            ])
        ], style={
            'textAlign': 'center',
            'padding': '40px',
            'color': '#f59e0b'
        })

    # Acheter des actions si le bouton est cliqué
    if buy_clicks:
        buy_stock(api_key='your_api_key', symbol=selected_stock, quantity=10)

    # Vendre des actions si le bouton est cliqué
    if sell_clicks:
        sell_stock(api_key='your_api_key', symbol=selected_stock, quantity=10)

    # Récupérer les informations de compte pour afficher les transactions en cours
    account_info = get_account_info(api_key='your_api_key')
    if account_info:
        transactions = account_info.get('transactions', [])
        df_transactions = pd.DataFrame(transactions)

        if df_transactions.empty:
            return html.Div([
                html.Span('📭', style={'fontSize': '2rem', 'marginBottom': '12px', 'display': 'block'}),
                html.Span('Aucune transaction trouvée', style={
                    'color': '#64748b',
                    'fontSize': '1rem'
                })
            ], style={'textAlign': 'center', 'padding': '40px'})

        # Créer une table HTML stylée
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th(col, style={
                        'padding': '12px 16px',
                        'textAlign': 'left',
                        'color': '#a78bfa',
                        'fontWeight': '600',
                        'fontSize': '0.8125rem',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.05em',
                        'borderBottom': '1px solid rgba(148, 163, 184, 0.1)'
                    }) for col in df_transactions.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df_transactions.iloc[i][col], style={
                        'padding': '12px 16px',
                        'color': '#f8fafc',
                        'fontSize': '0.9375rem',
                        'borderBottom': '1px solid rgba(148, 163, 184, 0.05)'
                    }) for col in df_transactions.columns
                ], style={
                    'transition': 'background-color 0.2s ease'
                }) for i in range(len(df_transactions))
            ])
        ], style={
            'width': '100%',
            'borderCollapse': 'collapse'
        })
    else:
        return html.Div([
            html.Span('📭', style={'fontSize': '2rem', 'marginBottom': '12px', 'display': 'block'}),
            html.Span('Aucune transaction trouvée', style={
                'color': '#64748b',
                'fontSize': '1rem'
            })
        ], style={'textAlign': 'center', 'padding': '40px'})
