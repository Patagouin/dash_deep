import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_base_data(start_date, end_date, open_price=100):
    # Générer une plage de dates pour les jours de bourse (lun-ven, 9h-17h)
    date_range = pd.date_range(start=start_date, end=end_date, freq='T')
    date_range = date_range[date_range.indexer_between_time('09:00', '17:00')]
    date_range = date_range[date_range.dayofweek < 5]  # Exclure les weekends
    
    # Créer un DataFrame avec une colonne 'price' initialisée à open_price
    df = pd.DataFrame(index=date_range, columns=['price'])
    df['price'] = open_price
    return df

def add_random_walk(df, volatility=0.001):
    # Ajouter un signal aléatoire (marche aléatoire)
    prices = [df.iloc[0]['price']]
    for _ in range(1, len(df)):
        change = np.random.normal(0, volatility)  # Changement aléatoire
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    df['price'] = prices
    return df

def add_trend(df, trend_strength=0.0001):
    # Ajouter une tendance linéaire faible
    trend = np.linspace(0, 1, len(df)) * trend_strength
    df['price'] *= (1 + trend)
    return df

def add_seasonality(df):
    # Ajouter une composante saisonnière (horaire et jour de la semaine)
    time_component = np.sin(2 * np.pi * df.index.hour / 24)
    day_component = np.sin(2 * np.pi * df.index.dayofweek / 7)
    seasonal_component = 0.01 * (time_component + day_component)
    df['price'] *= (1 + seasonal_component)
    return df

def add_lunch_effect(df, effect_strength=0.005):
    # Ajouter un effet de pause déjeuner (12h-14h)
    lunch_hours = (df.index.hour >= 12) & (df.index.hour < 14)
    lunch_effect = np.where(lunch_hours, -effect_strength, 0)
    df['price'] *= (1 + lunch_effect)
    return df

def generate_stock_data(start_date, end_date, data_type='random_walk', open_price=100):
    # Générer des données de cours boursier avec différents types de signaux
    df = generate_base_data(start_date, end_date, open_price)
    
    if data_type == 'random_walk':
        df = add_random_walk(df)
    elif data_type == 'trend':
        df = add_random_walk(df)
        df = add_trend(df)
    elif data_type == 'seasonal':
        df = add_random_walk(df)
        df = add_seasonality(df)
    elif data_type == 'lunch_effect':
        df = add_random_walk(df)
        df = add_lunch_effect(df)
    else:
        raise ValueError("Unsupported data type")
    
    return df

if __name__ == "__main__":
    # Test de génération de données
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    data_types = ['random_walk', 'trend', 'seasonal', 'lunch_effect']
    
    for data_type in data_types:
        df = generate_stock_data(start_date, end_date, data_type)
        print(f"\nData type: {data_type}")
        print(df.head())
        print(df.tail())