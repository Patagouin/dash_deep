# prediction_utils.py

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
import datetime
import utils as ut
import math
import random
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.optimizers import Adam

# # Les fonctions

# +
# dataX = look_back_x cotations séparé par stride
# dataY = nb_y points réparti équitableme
def create_X_Y(dataset, data_info):
    look_back_x = data_info['look_back_x']
    nb_quots_by_day = data_info['nb_quots_by_day']
    stride_x = data_info['stride_x']
    nb_y = data_info['nb_y']
    return_type = data_info.get('return_type', 'value')  # default to 'value' if not provided
    data_X, data_Y = [], []
    # Pour chaque point de départ de journée possible
    for i in range(0, len(dataset), nb_quots_by_day):
        # EX : journée 1 = data_X = data[0,2,4,...120]
        a = dataset.iloc[i:i + look_back_x*stride_x:stride_x, :]  # premières minutes de la journée
        if return_type == 'yield':
            a /= a.iloc[-1, :]  # divide by the last train value of the day dans le cas pourcentage
        data_X.append(a)
        y_values = []
        # ex: 540 quots - 60 * 2 // 2+1 = 420 / 3 = 140
        # (nb_quots_by_day - look_back_x*stride_x) = (on retire la quantité de data du début qui a servi à l'entrainement)
        # (nb_y+1) nb de Y par jour, +1 car ce que l'on veut c'est nb_y + 1 interval pour nb_y point qui ne soit ni au début ni à la fin
        stride_y = (nb_quots_by_day - look_back_x*stride_x) // (nb_y+1) # écart des Y pour une bonne répartition sur la journée
        # [140, 280]
        offsets = [stride_y*j for j in range(1,nb_y+1)] #  POsition des Y
        for offset in offsets:
            # [60*2 + 140, 60*2 + 280] = [260, 400]
            y_value = dataset.iloc[i + look_back_x*stride_x + offset, 0]  # On ne prend que les cotations car ca ne sert à rien de prédire un volume on veut juste une valeur
            if return_type == 'yield':
                y_value /= dataset.iloc[i+look_back_x*stride_x, 0]  # divide by the last train value of the day dans le cas pourcentage
            y_values.append(y_value)
        data_Y.append(y_values)
    return np.array(data_X), np.array(data_Y)

# Fonction pour créer les ensembles d'entraînement et de test
# Il faut que le dataset ait été cleaner (et surtout ne pas avoir pris les cotations de la derniere journée incomplete)
def split_dataset_train_test(dataset, data_info):
    shareObj = data_info['shareObj']
    nb_days_to_take_dataset = data_info['nb_days_to_take_dataset']
    percent_train_test = data_info['percent_train_test']
    
    # Déterminer le nombre de minutes jusqu'à l'ouverture et la fermeture du marché
    nb_minute_until_open = shareObj.openRichMarketTime.hour * 60 + shareObj.openRichMarketTime.minute
    nb_minute_until_close = shareObj.closeRichMarketTime.hour * 60 + shareObj.closeRichMarketTime.minute
    # Calculer le nombre de cotations par jour
    nb_quots_by_day = (nb_minute_until_close - nb_minute_until_open) + 1
    if (data_info['look_back_x']*data_info['stride_x'] < nb_quots_by_day):
        data_info['nb_quots_by_day'] = nb_quots_by_day
    else:
        print("look_back_x*stride_x > nb_quots_by_day")
        exit(1)

    nb_days_total = len(dataset) // nb_quots_by_day
    data_info['nb_days_total'] = nb_days_total
    
    if (nb_days_to_take_dataset == 'max' or nb_days_to_take_dataset > nb_days_total):
        # Calculer le nombre total de jours de cotations dans le dataset
        nb_days_to_take = nb_days_total
    else:
        nb_days_to_take = nb_days_to_take_dataset
        
    # Calculer le nombre de jours de cotations à sauter pour créer un ensemble de données d'entraînement équilibré
    # Car on ne prend que les jours de cotations les plus récents
    nb_days_skipped =  (nb_days_total - nb_days_to_take)

    nb_day_train = math.floor(nb_days_to_take * (percent_train_test/100))
    nb_day_test = nb_days_to_take - nb_day_train

    nb_quots_skipped = nb_days_skipped * nb_quots_by_day

    nb_quots_train = nb_day_train * nb_quots_by_day
    nb_quots_test = nb_day_test * nb_quots_by_day # Non utile car les quots de tests sont les quots restantes
    # Créer les ensembles d'entraînement et de test en fonction des tailles calculées
    train = dataset.iloc[nb_quots_skipped:nb_quots_skipped + nb_quots_train, :]
    test = dataset.iloc[nb_quots_skipped + nb_quots_train:len(dataset), :]
    return train, test

# Fonction pour créer les ensembles d'entraînement et de test avec un look_back spécifique
def create_train_test(dataset, data_info):
    train, test = split_dataset_train_test(dataset, data_info)
    trainX, trainY = create_X_Y(train, data_info)
    testX, testY = create_X_Y(test, data_info)
    return trainX, trainY, testX, testY

# Fonction pour déterminer la plage de dates et préparer les données de cotation
def get_and_clean_data(shM, shareObj, columns):
    # Si la cotation ne se termine pas en fin de journée alors on exclue cette journée pour avoir des journées complètes
    if shareObj.lastRecord.time() < shareObj.closeRichMarketTime:
        end_date = shareObj.lastRecord - datetime.timedelta(days=1)
        end_date = end_date.replace(hour=shareObj.closeRichMarketTime.hour, 
                                    minute=shareObj.closeRichMarketTime.minute, 
                                    second=shareObj.closeRichMarketTime.second)
    else:
        end_date = shareObj.lastRecord

    data_quots = shM.get_cotations_data_df(shareObj, shareObj.firstRecord, end_date)
    # On interpole les valeurs null
    df = ut.prepareData(shareObj, data_quots, columns)
    return df

def directional_accuracy(y_true, y_pred):
    # Assuming last_known_value is the first value in y_true
    last_known_value = y_true[:, 0]
    # Repeat last_known_value to match the shape of y_true
    last_known_value = tf.expand_dims(last_known_value, axis=-1)
    last_known_value = tf.tile(last_known_value, [1, y_true.shape[1]])
    true_direction = tf.math.sign(y_true - last_known_value)
    pred_direction = tf.math.sign(y_pred - last_known_value)

    correct_predictions = tf.cast(tf.equal(true_direction, pred_direction), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)
    # Custom run_trial method to choose between 'val_loss' and 'directional_accuracy'
    
class CustomHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # Choisir aléatoirement entre 'val_loss' et 'directional_accuracy'
        objective = random.choice(['val_loss', 'directional_accuracy'])
        self.oracle.objective = kt.Objective(objective, direction='min' if objective == 'val_loss' else 'max')
        results = super().run_trial(trial, *args, **kwargs)
        return results

class CustomRandomSearch(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Choisir aléatoirement entre 'val_loss' et 'directional_accuracy'
        objective = random.choice(['val_loss', 'directional_accuracy'])
        self.oracle.objective = objective
        super().run_trial(trial, *args, **kwargs)

# la fermeture permet d'obtenir une fonction avec la bonne signature
def build_model_closure(data_info, hps):

    def build_model(hp):
        model = keras.Sequential()
        num_layers = hp.Choice('num_layers', values=hps['layers'])
    
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=hps['nb_units'])
            if i == 0:
                model.add(keras.layers.LSTM(units, input_shape=(data_info['look_back_x'], len(data_info['features'])), return_sequences=(num_layers > 1)))
            else:
                model.add(keras.layers.LSTM(units, return_sequences=(i != num_layers - 1)))

        model.add(keras.layers.Dense(data_info['nb_y']))

        # Vous pouvez également optimiser le taux d'apprentissage
        optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values=hps['learning_rate']))
        return_type = data_info.get('return_type', 'yield') # par défaut yield
        if return_type == 'yield':
            model.compile(loss=hps['loss'], optimizer=optimizer, metrics=[directional_accuracy])
        else:
            model.compile(loss=hps['loss'], optimizer=optimizer)
        return model
    return build_model


def build_model_closure(data_info, hps):
    """
    Crée une fermeture pour construire un modèle Keras avec des hyperparamètres.

    @param data_info: Dictionnaire contenant des informations sur les données.
                      - 'look_back_x': Nombre de pas de temps à regarder en arrière.
                      - 'features': Liste des noms de caractéristiques.
                      - 'nb_y': Nombre de caractéristiques de sortie.
                      - 'return_type': Type de retour ('yield' ou 'value').
    @param hps: Dictionnaire contenant les choix d'hyperparamètres.
                - 'layers': Liste du nombre possible de couches.
                - 'nb_units': Liste du nombre possible d'unités par couche.
                - 'learning_rate': Liste des taux d'apprentissage possibles.
                - 'loss': Fonction de perte à utiliser.

    @return: Une fonction qui construit un modèle Keras avec les hyperparamètres donnés.
    """

    def build_model(hp):
        """
        Construit un modèle Keras basé sur les hyperparamètres donnés.

        @param hp: Objet d'hyperparamètres de Keras Tuner.

        @return: Modèle Keras compilé.
        """
        # Définir le tenseur d'entrée
        input_tensor = keras.layers.Input(shape=(data_info['look_back_x'], len(data_info['features'])))
        
        # Extraction de last_known_value via une couche Lambda
        aux_output = keras.layers.Lambda(lambda x: x[:,-1,0], name='aux_output')(input_tensor)
        
        # Initialiser x avec le tenseur d'entrée
        x = input_tensor
        
        # Ajouter des couches LSTM
        num_layers = hp.Choice('num_layers', values=hps['layers'])
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=hps['nb_units'])
            x = keras.layers.LSTM(units, return_sequences=(i != num_layers - 1))(x)
        
        # Ajouter la couche de sortie principale
        main_output = keras.layers.Dense(data_info['nb_y'], name='main_output')(x)
        
        # Créer le modèle en spécifiant les entrées et les sorties
        model = keras.models.Model(inputs=input_tensor, outputs=[main_output, aux_output])
        
        # Compiler le modèle
        optimizer = keras.optimizers.Adam(hp.Choice('learning_rate', values=hps['learning_rate']))
        return_type = data_info.get('return_type', 'value')
        
        if return_type == 'yield':
            print ('yield')
            model.compile(loss={'main_output': hps['loss'], 'aux_output': 'mse'}, optimizer=optimizer, metrics={'main_output': directional_accuracy})
											   
																		
        else:
            model.compile(loss={'main_output': hps['loss'], 'aux_output': 'mse'}, optimizer=optimizer)
        
        return model

    return build_model

def tuner_results_to_dataframe(tuner):
    # Extraction des métriques et des hyperparamètres des essais
    results = []
    
    # Parcourir tous les essais
    for trial in tuner.oracle.trials.values():
        trial_id = trial.trial_id
        
        # Récupération des métriques pour l'essai actuel
        all_metrics = list(trial.metrics.metrics.keys())
        print(f"Available metrics for trial {trial_id}: {all_metrics}")  # Pour déboguer

        # Récupération de directional_accuracy
        if 'directional_accuracy' in all_metrics:
            metrics_history = trial.metrics.get_history('directional_accuracy')
            val_dir_acc_best_value = max([entry.value for entry in metrics_history])
        else:
            val_dir_acc_best_value = None
            print(f"directional_accuracy not available for trial {trial_id}")
            
        # Récupération de val_loss
        if 'val_loss' in all_metrics:
            val_loss_best_value = trial.metrics.get_best_value('val_loss')
        else:
            val_loss_best_value = None
            print(f"val_loss not available for trial {trial_id}")
        
        # Ajout des métriques et des hyperparamètres à la liste des résultats
        results.append({
            'trial_id': trial_id,
            'directional_accuracy': val_dir_acc_best_value,
            'val_loss': val_loss_best_value,
            'hyperparameters': trial.hyperparameters.values
        })
    
    return pd.DataFrame(results)

def train_and_select_best_model(data_info, hps, trainX, trainY, testX, testY):
    """
    Entraîne des modèles avec plusieurs configurations et sélectionne le meilleur.

    Args:
        data_info (dict): Informations sur les données.
        hps (dict): Hyperparamètres à tester.
        trainX, trainY: Données d'entraînement.
        testX, testY: Données de validation.

    Returns:
        best_model: Le modèle entraîné avec les meilleures performances.
        best_history: Historique d'entraînement du meilleur modèle.
    """

    # Construire le modèle à l'aide de Keras Tuner
    build_model = build_model_closure(data_info, hps)

    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='tune_model'
    )

    tuner.search(trainX, trainY, epochs=hps['epochs_tuner'], validation_data=(testX, testY))

    # Obtenir les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Construire le modèle avec les meilleurs hyperparamètres et l'entraîner
    best_model = build_model(best_hps)
    best_history = best_model.fit(trainX, trainY, epochs=hps['epochs'], validation_data=(testX, testY))

    return best_model, best_history

def train_with_hyperband(data_info, hps, trainX, trainY, testX, testY):
    """
    Utilise un LSTM avec la stratégie Hyperband pour trouver les meilleurs hyperparamètres.

    Args:
        data_info (dict): Informations sur les données.
            - 'look_back_x': Nombre de pas de temps à regarder en arrière.
            - 'features': Liste des noms de caractéristiques.
            - 'nb_y': Nombre de caractéristiques de sortie.
            - 'return_type': Type de retour ('yield' ou 'value').
        hps (dict): Hyperparamètres à tester.
            - 'nb_units': Liste du nombre possible d'unités par couche.
            - 'layers': Liste du nombre possible de couches.
            - 'learning_rate': Liste des taux d'apprentissage possibles.
            - 'loss': Fonction de perte à utiliser.
            - 'epochs_tuner': Nombre d'époques pour le tuner.
            - 'batch_size': Taille des lots.
        trainX, trainY: Données d'entraînement.
        testX, testY: Données de validation.

    Returns:
        best_model: Le modèle entraîné avec les meilleures performances.
        best_hps: Les meilleurs hyperparamètres trouvés.
        tuner: L'objet tuner contenant les résultats de la recherche.
    """

    def build_model(hp):
        """
        Construit un modèle LSTM basé sur les hyperparamètres donnés.

        Args:
            hp: Objet d'hyperparamètres de Keras Tuner.

        Returns:
            Un modèle Keras compilé.
        """
        input_tensor = Input(shape=(data_info['look_back_x'], len(data_info['features'])))
        
        # Extraction de last_known_value via une couche Lambda
        aux_output = Lambda(lambda x: x[:, -1, 0], name='aux_output')(input_tensor)
        
        x = input_tensor
        num_layers = hp.Choice('num_layers', values=hps['layers'])
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=hps['nb_units'])
            x = LSTM(units, return_sequences=(i != num_layers - 1))(x)
        
        main_output = Dense(data_info['nb_y'], name='main_output')(x)
        
        model = Model(inputs=input_tensor, outputs=[main_output, aux_output])
        
        optimizer = Adam(hp.Choice('learning_rate', values=hps['learning_rate']))
        return_type = data_info.get('return_type', 'value')
        
        if return_type == 'yield':
            model.compile(
                loss={'main_output': hps['loss'], 'aux_output': 'mse'},
                optimizer=optimizer,
                metrics={'main_output': directional_accuracy}
            )
        else:
            model.compile(
                loss={'main_output': hps['loss'], 'aux_output': 'mse'},
                optimizer=optimizer
            )
        
        return model

    # Configurer le tuner Hyperband
    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=hps['epochs_tuner'],
        factor=3,
        directory='hyperband_dir',
        project_name='lstm_hyperband'
    )

    # Définir un callback pour arrêter tôt si les performances ne s'améliorent pas
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Lancer la recherche d'hyperparamètres
    tuner.search(
        trainX, trainY,
        validation_data=(testX, testY),
        epochs=hps['epochs_tuner'],
        batch_size=hps['batch_size'],
        callbacks=[stop_early]
    )

    # Obtenir les meilleurs hyperparamètres
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Construire le modèle avec les meilleurs hyperparamètres
    best_model = tuner.hypermodel.build(best_hps)

    # Entraîner le modèle avec les meilleurs hyperparamètres
    history = best_model.fit(
        trainX, trainY,
        validation_data=(testX, testY),
        epochs=hps['epochs'],
        batch_size=hps['batch_size'],
        callbacks=[stop_early]
    )

    return best_model, best_hps, tuner
