import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models import create_lstm_model, create_gru_model, create_transformer_model  # Added import for models

def prepare_data(df, lookback=60, forecast_horizon=5):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['price']])
    
    X, y = [], []
    for i in range(len(scaled_data) - lookback - forecast_horizon):
        X.append(scaled_data[i:i+lookback])
        y.append(scaled_data[i+lookback:i+lookback+forecast_horizon])
    
    return np.array(X), np.array(y), scaler

def train_model(model, X_train, y_train, epochs, batch_size, learning_rate, loss):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return history

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and true values
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    mse = np.mean((predictions - y_test)**2)
    mae = np.mean(np.abs(predictions - y_test))
    
    return mse, mae, predictions

def run_experiment(df, model_type, num_layers, units=None, learning_rate=0.001, loss='mse', epochs=50, batch_size=32, d_model=None, num_heads=None, dff=None):
    X, y, scaler = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    
    if model_type == 'LSTM':
        model = create_lstm_model(input_shape, output_shape, num_layers, units)
    elif model_type == 'GRU':
        model = create_gru_model(input_shape, output_shape, num_layers, units)
    elif model_type == 'Transformer':
        model = create_transformer_model(input_shape, output_shape, num_layers, d_model, num_heads, dff)
    else:
        raise ValueError("Unsupported model type")
    
    history = train_model(model, X_train, y_train, epochs, batch_size, learning_rate, loss)
    mse, mae, predictions = evaluate_model(model, X_test, y_test, scaler)
    
    return model, history, mse, mae, predictions