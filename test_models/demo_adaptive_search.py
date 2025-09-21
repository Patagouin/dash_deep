"""
Démonstration de la recherche adaptative d'hyperparamètres pour modèles LSTM
---------------------------------------------------------------------------

Ce script montre comment utiliser l'approche en deux phases pour l'optimisation:
1. Exploration large avec RandomSearch
2. Raffinement avec Hyperband sur les régions prometteuses

Cette approche permet de créer des modèles pour:
- Chaque action individuelle
- Chaque secteur (groupes d'actions)
- Un modèle global pour toutes les actions
"""

import sys
sys.path.append('..')

# Importations standards
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt
from datetime import time

# Importations du module create_model
from create_model import (
    get_and_clean_data, 
    create_train_test, 
    train_with_adaptive_search,
    create_models_hierarchy,
    save_models_to_database,
    tuner_results_to_dataframe
)

def main():
    """Fonction principale de démonstration"""
    print("Démonstration de la recherche adaptative d'hyperparamètres pour modèles LSTM")
    print("=" * 80)

    # 1. Configuration des données et des hyperparamètres
    print("\n1. Configuration des données et des hyperparamètres")
    print("-" * 50)
    
    data_info = {
        'look_back_x': 30,            # Nombre de points de données historiques
        'stride_x': 2,                # Pas entre les points de données
        'nb_y': 2,                    # Nombre de points de prédiction
        'nb_days_to_take_dataset': 60, # Nombre de jours à considérer
        'percent_train_test': 0.8,    # Pourcentage des données pour l'entraînement
        'features': ['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume'],
        'return_type': 'yield'        # Type de retour: 'yield' ou 'value'
    }
    
    hyperparameters = {
        'nb_units': [32, 64, 128, 256, 512],   # Nombre d'unités par couche
        'layers': [1, 2, 3],                   # Nombre de couches LSTM
        'learning_rate': [0.001, 0.005, 0.01], # Taux d'apprentissage
        'loss': 'mse',                         # Fonction de perte
        'batch_size': 32,                      # Taille des lots
        'epochs_tuner': 50                     # Nombre d'époques pour le tuner
    }

    print("Configuration des données:")
    for key, value in data_info.items():
        print(f"  - {key}: {value}")
    
    print("\nConfiguration des hyperparamètres:")
    for key, value in hyperparameters.items():
        print(f"  - {key}: {value}")

    # 2. Génération de données simulées
    print("\n2. Génération de données simulées")
    print("-" * 50)
    
    # Simuler une classe ShareManager
    class ShareManager:
        def __init__(self, symbol, sector=None):
            self.symbol = symbol
            self.sector = sector
            self.openRichMarketTime = time(9, 0)  # 9:00 AM
            self.closeRichMarketTime = time(17, 30)  # 5:30 PM

    # Définir quelques actions et leurs secteurs
    stock_sectors = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'JPM': 'Finance',
        'BAC': 'Finance',
        'XOM': 'Energy',
        'CVX': 'Energy'
    }
    
    # Générer les données pour chaque action
    share_data_dict = {}
    for symbol, sector in stock_sectors.items():
        data = generate_stock_data(symbol)
        share_manager = ShareManager(symbol, sector)
        share_data_dict[symbol] = (data, share_manager)

    print(f"Données générées pour {len(share_data_dict)} actions")

    # 3. Sélection d'une action pour le test
    print("\n3. Test de la recherche adaptative sur une action")
    print("-" * 50)
    
    test_symbol = 'AAPL'
    print(f"Action sélectionnée pour le test: {test_symbol}")
    
    test_data, test_shareObj = share_data_dict[test_symbol]
    
    # Préparer les données pour cette action
    test_data_info = data_info.copy()
    test_data_info['shareObj'] = test_shareObj
    
    # Nettoyer les données
    print("Nettoyage des données...")
    clean_data = get_and_clean_data(test_data, test_shareObj, test_data_info['features'])
    
    # Créer les ensembles d'entraînement et de test
    print("Création des ensembles d'entraînement et de test...")
    trainX, trainY, testX, testY = create_train_test(clean_data, test_data_info)
    
    print(f"Formes des données d'entraînement: X: {trainX.shape}, Y: {trainY.shape}")
    print(f"Formes des données de test: X: {testX.shape}, Y: {testY.shape}")
    
    # Réduire les hyperparamètres pour une démo plus rapide
    test_hps = hyperparameters.copy()
    test_hps['epochs_tuner'] = 3  # Très peu d'époques pour la démo
    test_hps['nb_units'] = [32, 64]  # Options réduites
    test_hps['layers'] = [1, 2]  # Options réduites
    
    # Option pour exécuter la recherche adaptative (commentée par défaut car lente)
    do_search = False  # Mettre à True pour exécuter la recherche
    
    if do_search:
        print("\nDémarrage de la recherche adaptative d'hyperparamètres...")
        model, best_hps, results = train_with_adaptive_search(
            test_data_info, test_hps, trainX, trainY, testX, testY
        )
        
        print("\nMeilleurs hyperparamètres trouvés:")
        for param, value in best_hps.values.items():
            print(f"  - {param}: {value}")
    else:
        print("\nLa recherche adaptative est désactivée dans cette démo.")
        print("Pour l'activer, définissez do_search = True")

    # 4. Exemple d'utilisation de create_models_hierarchy
    print("\n4. Exemple d'utilisation de create_models_hierarchy")
    print("-" * 50)
    
    # Pour l'exemple, utiliser un sous-ensemble d'actions
    demo_share_data = {k: share_data_dict[k] for k in list(share_data_dict.keys())[:3]}
    demo_sectors = {k: stock_sectors[k] for k in demo_share_data.keys()}
    
    print(f"Utilisation d'un sous-ensemble de {len(demo_share_data)} actions pour la démo")
    
    # Réduire encore plus les hyperparamètres pour une démo plus rapide
    demo_hps = hyperparameters.copy()
    demo_hps['epochs_tuner'] = 2  # Très peu d'époques pour la démo
    demo_hps['nb_units'] = [32]  # Options minimales
    demo_hps['layers'] = [1]  # Options minimales
    
    # Option pour exécuter la création de hiérarchie de modèles
    do_hierarchy = False  # Mettre à True pour exécuter
    
    if do_hierarchy:
        print("\nDémarrage de la création de la hiérarchie de modèles...")
        models_dict, metrics_dict = create_models_hierarchy(
            demo_share_data, demo_sectors, data_info, demo_hps
        )
        
        print("\nModèles créés:")
        print(f"  - Modèles individuels: {len(models_dict['individual'])}")
        print(f"  - Modèles sectoriels: {len(models_dict['sector'])}")
        print(f"  - Modèle global: {'Oui' if models_dict['global'] is not None else 'Non'}")
        
        # 5. Exemple d'utilisation de save_models_to_database
        print("\n5. Exemple d'utilisation de save_models_to_database")
        print("-" * 50)
        
        db_config = {
            'type': 'sqlite',
            'path': 'demo_models_database.db'
        }
        
        print(f"Sauvegarde des modèles dans la base de données {db_config['path']}...")
        save_success = save_models_to_database(models_dict, metrics_dict, db_config)
        
        if save_success:
            print("Sauvegarde des modèles terminée avec succès")
    else:
        print("\nLa création de la hiérarchie de modèles est désactivée dans cette démo.")
        print("Pour l'activer, définissez do_hierarchy = True")

    print("\nDémonstration terminée!")
    print("=" * 80)

def generate_stock_data(symbol, days=60, nb_quots_by_day=510):
    """Fonction pour générer des données simulées d'actions"""
    np.random.seed(int(hash(symbol) % 1000))  # Seed basé sur le symbole
    
    # Prix initial et volatilité basés sur le symbole
    base_price = 100 + hash(symbol) % 400
    volatility = 0.01 + 0.02 * (hash(symbol) % 5) / 5
    
    # Générer prix de clôture avec marche aléatoire
    total_points = days * nb_quots_by_day
    returns = np.random.normal(0, volatility, total_points)
    price_series = base_price * np.exp(np.cumsum(returns))
    
    # Générer OHLC et volume
    data = []
    for i in range(total_points):
        close = price_series[i]
        daily_volatility = volatility * close
        open_price = close + np.random.normal(0, daily_volatility)
        high_price = max(open_price, close) + abs(np.random.normal(0, daily_volatility))
        low_price = min(open_price, close) - abs(np.random.normal(0, daily_volatility))
        volume = int(np.random.gamma(2.0, 100000) * (1 + np.sin(i/100) * 0.2))
        
        data.append([open_price, high_price, low_price, close, volume])
    
    # Créer DataFrame
    df = pd.DataFrame(data, columns=['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume'])
    return df

if __name__ == "__main__":
    main() 