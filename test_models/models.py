import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Input, MultiHeadAttention, LayerNormalization, Dropout

def create_lstm_model(input_shape, output_shape, num_layers, units):
    # Créer un modèle LSTM
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(LSTM(units[i], return_sequences=return_sequences))
    
    model.add(Dense(output_shape))
    return model

def create_gru_model(input_shape, output_shape, num_layers, units):
    # Créer un modèle GRU
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(GRU(units[i], return_sequences=return_sequences))
    
    model.add(Dense(output_shape))
    return model

def create_transformer_model(input_shape, output_shape, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
    # Créer un modèle Transformer
    inputs = Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_layers):
        # Multi-Head Attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attention_output = Dropout(dropout_rate)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed Forward Network
        ffn_output = Dense(dff, activation="relu")(x)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    outputs = Dense(output_shape)(x)
    
    return Model(inputs=inputs, outputs=outputs)