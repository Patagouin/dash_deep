from data_generation import generate_stock_data
from train_and_evaluate import run_experiment
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stock_data_interactive(df, title="Stock Price Data"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_prediction_interactive(true_data, pred_data, title="Stock Price Prediction"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=true_data.index, y=true_data['price'], mode='lines', name='True'))
    fig.add_trace(go.Scatter(x=pred_data.index, y=pred_data['price'], mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig

if __name__ == "__main__":
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    data_types = ['random_walk', 'trend', 'seasonal', 'lunch_effect']
    
    # Générer et afficher les données pour chaque type
    for data_type in data_types:
        print(f"\nGenerating {data_type} data")
        df = generate_stock_data(start_date, end_date, data_type)
        
        # Visualiser les données de manière interactive
        fig = plot_stock_data_interactive(df, f"Generated Stock Data - {data_type}")
        fig.show()
    
    # Attendre que l'utilisateur appuie sur Entrée pour continuer
    input("Données générées et affichées. Appuyez sur Entrée pour continuer avec les expériences...")
    
    # Le reste du code pour les expériences
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
            
            # Visualiser les prédictions de manière interactive
            fig = plot_prediction_interactive(df[-len(predictions):], predictions, f"Predictions for {exp['model_type']} - {data_type}")
            fig.show()
        
        # Afficher les résultats
        for result in results:
            print(f"Experiment: {result['experiment']}")
            print(f"MSE: {result['mse']:.4f}")
            print(f"MAE: {result['mae']:.4f}")
            print("---")