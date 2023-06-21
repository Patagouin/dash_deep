import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import Models.SqlCom as sq
import Models.utils as ut
import Models.Shares as sm
import datetime
import math
import matplotlib.pyplot as plt

#def create_dataset(data, look_back, prediction_horizon):
#    X, y = [], []
#    for i in range(len(data) - look_back - prediction_horizon):
#        X.append(data[i:(i + look_back), 0])
#        y.append(data[(i + look_back):(i + look_back + prediction_horizon), 0])
#    return np.array(X), np.array(y)

#def create_dataset(dataset, look_back):
#    data_X, data_Y = [], []
#    for i in range(0,len(dataset) - look_back, look_back):
#        a = dataset[i:(i + look_back), :]
#        data_X.append(a)
#        data_Y.append(dataset[i + look_back, 0])
#    return np.array(data_X), np.array(data_Y)


def create_dataset(dataset, look_back, nb_quots_by_day):
    data_X, data_Y = [], []
    for i in range(0,len(dataset) - 2*look_back, 1):
       # if (i//nb_quots_by_day ==  (i+2*look_back)//nb_quots_by_day ): # Il faudrait éviter la fin de journée le probleme 

        a = dataset[i:(i + look_back), :]
        data_X.append(a)
        b = dataset[(i + look_back):(i + 2*look_back), :]
        data_Y.append(b)
    return np.array(data_X), np.array(data_Y)

def test_lstm(shareObj, data_quots):
    df = ut.prepareData(shareObj, data_quots) #column="openPrice"

    np.random.seed(42)


    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df) 

    # Split the dataset into training and testing sets
    nb_minute_until_open = shareObj.openRichMarketTime.hour * 60 + shareObj.openRichMarketTime.minute
    nb_minute_until_close = shareObj.closeRichMarketTime.hour * 60 + shareObj.closeRichMarketTime.minute
    nb_quots_by_day = (nb_minute_until_close - nb_minute_until_open) + 1 # +1 interval inclus
    nb_days_quots_total = len(dataset) // nb_quots_by_day
    nb_days_quots_train = math.floor(nb_days_quots_total) - 1
    train_size = nb_days_quots_train * nb_quots_by_day
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    look_back = 60

    # Prepare the data with a look_back
    trainX, trainY = create_dataset(train, look_back, nb_quots_by_day)
    testX, testY = create_dataset(test, look_back, nb_quots_by_day)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], -1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], -1))

    # Create lists to store the error values
    train_scores = []
    test_scores = []
    figures = []  # Liste pour stocker les figures

    # Create and fit the LSTM model
    model = Sequential()
    model.add(LSTM(look_back, input_shape=(look_back, 1), return_sequences=True))
    model.add(Dense(look_back))
    model.add(LSTM(look_back))
    model.add(Dense(look_back))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Loop over epochs and track the error values
    trainYInv = scaler.inverse_transform(trainY.reshape(trainY.shape[0], -1))
    testYInv = scaler.inverse_transform(testY.reshape(testY.shape[0], -1))

    epochs = 2
    for epoch in range(epochs):
        model.fit(trainX, trainY, epochs=1, batch_size=nb_quots_by_day, verbose=2)
    
        # Make predictions
        trainPredict = model.predict(trainX)
        #testPredict = model.predict(testX)
    
        # Invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        #testPredict = scaler.inverse_transform(testPredict)
    
        # Calculate root mean squared error
        trainScore = np.sqrt(mean_squared_error(trainYInv, trainPredict))
        #testScore = np.sqrt(mean_squared_error(testYInv, testPredict))
    
        # Store the error values
        train_scores.append(trainScore)
        #test_scores.append(testScore)
    
        print(f'Epoch {epoch+1}: Train Score = {trainScore:.2f} RMSE')

    # Make predictions
    testPredict = testX[0] # [nb_quots_day,loop_back,1]

    #testInstance = np.expand_dims(testX[0][i:i+look_back], axis=0)  # L'entrée doit être de dimension [samples, time steps, features]
    #predictInstance = model.predict(testInstance)
    #predictInstance = np.reshape(predictInstance[0], (1, 1))
    #testPredict = np.append(testPredict, predictInstance, axis=0)

    for i in range(0,nb_quots_by_day - 2*look_back, look_back): #for i in range(0,nb_quots_by_day - 2*look_back, ):
        testInstance = np.expand_dims(testPredict[i:i+look_back], axis=0)  # L'entrée doit être de dimension [samples, time steps, features]
        predictInstance = model.predict(testInstance)
        predictInstance = np.reshape(predictInstance[0], (look_back, 1)) # predictInstance = np.reshape(predictInstance[0], (1, 1))
        testPredict = np.concatenate((testPredict, predictInstance), axis=0) #np.append(testPredict, predictInstance, axis=0)

    testPredict = np.array(testPredict).reshape(-1, 1)  # Convertir la liste en numpy array et redimensionner
    testPredict = scaler.inverse_transform(testPredict)  # Inverser la normalisation

    #testPredict = model.predict(testX)
    #testPredict = scaler.inverse_transform(testPredict)
    # Rajout des loop_back valeurs de testX
    testX_inv = scaler.inverse_transform(testX[0])
    #testYInv = np.concatenate((testX_inv, testYInv), axis=0)

    testScore = np.sqrt(mean_squared_error(testYInv, testPredict))
    print(f'Epoch {epoch+1}: Test Score = {testScore:.2f} RMSE')

    plt.plot(testYInv, label='Actual')
    plt.plot(testPredict, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Plot the error values
    plt.plot(train_scores, label='Train Score')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()

    return model, trainScore, testScore


# La difference avec la fonction de test au dessus est l'ensemble de train qui prend toutes les cotations
def compute_lstm(shareObj, data_quots):
    df = ut.prepareData(shareObj, data_quots) #column="openPrice"

    np.random.seed(42)

    look_back = 1

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df) 
    #df['Open'].values.reshape(-1, 1))
    #dataset = df.values
    #dataset[:, 0] = scaler.fit_transform(df['openPrice'].values.reshape(-1, 1)).flatten()

    # Split the dataset into training and testing sets
    nb_minute_until_open = shareObj.openRichMarketTime.hour * 60 + shareObj.openRichMarketTime.minute
    nb_minute_until_close = shareObj.closeRichMarketTime.hour * 60 + shareObj.closeRichMarketTime.minute
    nb_minute_quot_duration = (nb_minute_until_close - nb_minute_until_open)
    nb_days_quots_total = len(dataset) // nb_minute_quot_duration
    nb_days_quots_train = nb_days_quots_total
    train_size = nb_days_quots_train * nb_minute_quot_duration
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Prepare the data with a look_back
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], -1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], -1))

    # Create and fit the LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=10, batch_size=nb_minute_quot_duration, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY.reshape(-1, 1))

	# Calculate root mean squared error
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict)) #trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

    
    print(f'Train Score: {trainScore:.2f} RMSE')
    testScore = np.sqrt(mean_squared_error(testY, testPredict)) #testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print(f'Test Score: {testScore:.2f} RMSE')

    return model, trainScore, testScore
