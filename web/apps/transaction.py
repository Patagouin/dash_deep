from dash import dcc, html
from dash.dependencies import Input, Output, State
from web.components.navigation import create_navigation  # Importer le composant de navigation
import pandas as pd
from Models.trading212 import buy_stock, sell_stock, get_account_info
from app import app, shM

# Layout de la page Transaction
layout = html.Div([
    html.H3('Transactions'),

    # Dropdown pour sélectionner les actions
    dcc.Dropdown(
        id='transaction_dropdown',
        options=[
            {'label': '{}'.format(stock.symbol), 'value': stock.symbol} 
            for stock in sorted(shM.dfShares.itertuples(), key=lambda stock: stock.symbol)
        ],
        multi=False,
        style={'width': '50%', 'margin': '10px auto'}
    ),

    # Boutons pour acheter et vendre des actions
    html.Div([
        html.Button('Buy', id='buy_button', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Sell', id='sell_button', n_clicks=0)
    ], style={'textAlign': 'center', 'margin': '20px 0'}),

    # Table pour afficher les transactions en cours
    html.Div(id='transaction_table'),

    # Navigation standardisée
    create_navigation()
], style={'backgroundColor': 'black', 'minHeight': '100vh'})

# Callback pour gérer les achats et ventes
@app.callback(
    Output('transaction_table', 'children'),
    Input('buy_button', 'n_clicks'),
    Input('sell_button', 'n_clicks'),
    State('transaction_dropdown', 'value')
)
def handle_transaction(buy_clicks, sell_clicks, selected_stock):
    if not selected_stock:
        return html.Div("Please select a stock.", style={'color': 'red'})

    # Acheter des actions si le bouton est cliqué
    if buy_clicks:
        buy_stock(api_key='your_api_key', symbol=selected_stock, quantity=10)  # Exemple d'achat de 10 actions

    # Vendre des actions si le bouton est cliqué
    if sell_clicks:
        sell_stock(api_key='your_api_key', symbol=selected_stock, quantity=10)  # Exemple de vente de 10 actions

    # Récupérer les informations de compte pour afficher les transactions en cours
    account_info = get_account_info(api_key='your_api_key')
    if account_info:
        transactions = account_info.get('transactions', [])
        df_transactions = pd.DataFrame(transactions)

        # Créer une table HTML pour afficher les transactions
        return html.Table([
            html.Thead(html.Tr([html.Th(col) for col in df_transactions.columns])),
            html.Tbody([
                html.Tr([html.Td(df_transactions.iloc[i][col]) for col in df_transactions.columns])
                for i in range(len(df_transactions))
            ])
        ], style={'width': '80%', 'margin': '20px auto', 'color': 'white'})
    else:
        return html.Div("No transactions found.", style={'color': 'red'}) 