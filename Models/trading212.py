import requests
import logging

# URL de base de l'API Trading 212 (à adapter selon la documentation officielle)
BASE_URL = "https://api.trading212.com"

# Fonction pour récupérer les informations de compte
def get_account_info(api_key):
    url = f"{BASE_URL}/account"
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching account info: {e}")
        return None

# Fonction pour acheter des actions
def buy_stock(api_key, symbol, quantity):
    url = f"{BASE_URL}/order"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "symbol": symbol,
        "quantity": quantity,
        "action": "BUY"
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error buying stock {symbol}: {e}")
        return None

# Fonction pour vendre des actions
def sell_stock(api_key, symbol, quantity):
    url = f"{BASE_URL}/order"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        "symbol": symbol,
        "quantity": quantity,
        "action": "SELL"
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error selling stock {symbol}: {e}")
        return None 