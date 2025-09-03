# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# %%
from data_generation import generate_stock_data
from visualization import plot_stock_data, plot_prediction
from train_and_evaluate import run_experiment
from datetime import datetime
import matplotlib.pyplot as plt

# %%
if __name__ == "__main__":
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    data_types = ['random_walk', 'trend', 'seasonal', 'lunch_effect']
    
    # Générer et afficher les données pour chaque type
    for data_type in data_types:
        print(f"\nGenerating {data_type} data")
        df = generate_stock_data(start_date, end_date, data_type)
        
        # Visualiser les données
        plot_stock_data(df, f"Generated Stock Data - {data_type}")
        plt.savefig(f"stock_data_{data_type}.png")
        plt.close()
    
    # Attendre que l'utilisateur appuie sur Entrée pour continuer
    input("Données générées et affichées. Appuyez sur Entrée pour continuer avec les expériences...")
    
    # Le reste du code pour les expériences reste inchangé
    for data_type in data_types:
        print(f"\nTesting {data_type} data")
        df = generate_stock_data(start_date, end_date, data_type)
        
        experiments = [
            {'model_type': 'LSTM', 'num_layers': 2, 'units': [64, 32], 'learning_rate': 0.001, 'loss': 'mse', 'epochs': 50, 'batch_size': 32},
            {'model_type': 'GRU', 'num_layers': 2, 'units': [64, 32], 'learning_rate': 0.001, 'loss': 'mse', 'epochs': 50, 'batch_size': 32},
            {'model_type': 'Transformer', 'num_layers': 2, 'd_model': 64, 'num_heads': 4, 'dff': 256, 'learning_rate': 0.001, 'loss': 'mse', 'epochs': 50, 'batch_size': 32},
        ]
        
        results = []
        
        for exp in experiments:
            print(f"Running experiment: {exp}")
            model, history, mse, mae, predictions = run_experiment(df, **exp)
            results.append({
                'experiment': exp,
                'mse': mse,
                'mae': mae
            })
            
            # Visualiser les prédictions
            plot_prediction(df[-len(predictions):], predictions, f"Predictions for {exp['model_type']} - {data_type}")
            plt.savefig(f"predictions_{data_type}_{exp['model_type']}.png")
            plt.close()
        
        # Afficher les résultats
        for result in results:
            print(f"Experiment: {result['experiment']}")
            print(f"MSE: {result['mse']:.4f}")
            print(f"MAE: {result['mae']:.4f}")
            print("---")
