import matplotlib.pyplot as plt
import seaborn as sns

def plot_stock_data(df, title="Stock Price Data"):
    # Visualiser les données de prix des actions
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['price'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_prediction(true_data, pred_data, title="Stock Price Prediction"):
    # Visualiser les prédictions par rapport aux données réelles
    plt.figure(figsize=(15, 7))
    plt.plot(true_data.index, true_data['price'], label='True')
    plt.plot(pred_data.index, pred_data['price'], label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()