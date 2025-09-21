# prediction_utils.py

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
import datetime
from . import utils as ut
import math
import random
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Lambda, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History

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
        a_df = dataset.iloc[i:i + look_back_x*stride_x:stride_x, :].copy()
        # Convertir explicitement en tableau numérique (shape: [look_back_x, nb_features])
        a_arr = a_df.to_numpy(dtype=float, copy=True)
        if return_type == 'yield':
            # Normaliser uniquement openPrice (colonne 0) par la valeur de base de fin de fenêtre
            base_price = a_arr[-1, 0] if a_arr.shape[0] > 0 else None
            if base_price is None or base_price == 0:
                # fenêtre invalide si base non définie ou nulle
                continue
            a_arr[:, 0] = a_arr[:, 0] / base_price
            # On laisse volume (colonne 1) tel quel; si besoin, appliquer un scaling amont (log1p/MinMax)
        data_X.append(a_arr)
        y_values = []
        # ex: 540 quots - 60 * 2 // 2+1 = 420 / 3 = 140
        # (nb_quots_by_day - look_back_x*stride_x) = (on retire la quantité de data du début qui a servi à l'entrainement)
        # (nb_y+1) nb de Y par jour, +1 car ce que l'on veut c'est nb_y + 1 interval pour nb_y point qui ne soit ni au début ni à la fin
        stride_y = (nb_quots_by_day - look_back_x*stride_x) // (nb_y+1) # écart des Y pour une bonne répartition sur la journée
        # [140, 280]
        offsets = [stride_y*j for j in range(1,nb_y+1)] #  POsition des Y
        for offset in offsets:
            # [60*2 + 140, 60*2 + 280] = [260, 400]
            y_price = dataset.iloc[i + look_back_x*stride_x + offset, 0]
            if return_type == 'yield':
                base = dataset.iloc[i+look_back_x*stride_x, 0]
                y_value = (y_price / base) if base not in [0, None] else 0.0
            else:
                y_value = y_price
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
    nb_quots_by_day = max(1, (nb_minute_until_close - nb_minute_until_open) + 1)
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
    train = dataset.iloc[nb_quots_skipped:nb_quots_skipped + nb_quots_train, :].copy()
    test = dataset.iloc[nb_quots_skipped + nb_quots_train:len(dataset), :].copy()
    # Nettoyer NaN/Inf
    train = train.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    test = test.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return train, test

# Fonction pour créer les ensembles d'entraînement et de test avec un look_back spécifique
def create_train_test(dataset, data_info):
    train, test = split_dataset_train_test(dataset, data_info)
    trainX, trainY = create_X_Y(train, data_info)
    testX, testY = create_X_Y(test, data_info)
    # Supprimer les séquences contenant des NaN
    def drop_nan_sequences(X, Y):
        if X.size == 0:
            return X, Y
        # Ne filtrer que sur la première feature (supposée être 'openPrice')
        mask = ~np.isnan(X[:, :, 0]).any(axis=1)
        X_clean = X[mask]
        Y_clean = Y[mask] if Y is not None and len(Y) == len(mask) else Y
        return X_clean, Y_clean
    trainX, trainY = drop_nan_sequences(trainX, trainY)
    testX, testY = drop_nan_sequences(testX, testY)
    # Propager info de jour
    data_info['nb_days_total'] = max(0, len(dataset) // data_info.get('nb_quots_by_day', 1))
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
    # Préparer/interpoler et aligner (minutely)
    df = ut.prepareData(shareObj, data_quots, columns)
    # Appliquer log1p sur volume s'il est présent pour conserver les zéros et réduire l'échelle
    if 'volume' in df.columns:
        import numpy as np
        df['volume'] = np.log1p(df['volume'].astype(float).clip(lower=0))
    # Assainir: remplacer Inf/−Inf puis dropna
    import numpy as np
    before_rows = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    after_rows = len(df)
    if before_rows != after_rows:
        import logging
        logging.info(f"[CLEAN] Dropped {before_rows - after_rows} rows due to NaN/Inf after prepareData")
    # Valider les données
    if df is None or df.empty:
        raise ValueError(f"Données insuffisantes pour {shareObj.symbol}: dataframe vide après préparation")
    # Au moins quelques valeurs non-NaN
    if df[columns].isna().all().all():
        raise ValueError(f"Données insuffisantes pour {shareObj.symbol}: toutes les valeurs sont NaN")
    return df

def directional_accuracy(y_true, y_pred):
    # Pour un retour de type 'yield', la baseline (dernier connu) est 1.0
    baseline = tf.ones_like(y_true)
    true_direction = tf.math.sign(y_true - baseline)
    pred_direction = tf.math.sign(y_pred - baseline)
    correct_predictions = tf.cast(tf.equal(true_direction, pred_direction), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

def make_profit_metric(k_trades: int, trade_volume: float):
    """Crée un métrique 'profit' moyen par batch basé sur le top-K offsets prédits.
    Hypothèse: y représente des rendements relatifs (>1 = gain) entre t0 et offset.
    On sélectionne les K offsets aux y_pred les plus élevés et on mesure le profit
    réel sur y_true pour ces offsets (capped à 0 pour éviter les pertes), multiplié par le volume.
    """
    k = max(1, int(k_trades or 1))
    vol = float(trade_volume or 1.0)

    def profit_metric(y_true, y_pred):
        import numpy as _np
        def _np_profit(y_t, y_p):
            # y_t, y_p: (batch, nb_y)
            if y_t.ndim == 1:
                y_t = y_t[_np.newaxis, :]
            if y_p.ndim == 1:
                y_p = y_p[_np.newaxis, :]
            profits = []
            for i in range(y_t.shape[0]):
                gains = _np.asarray(y_t[i], dtype=_np.float32) - 1.0
                preds = _np.asarray(y_p[i], dtype=_np.float32)
                order = _np.argsort(-preds)
                sel = order[:k]
                realized = _np.maximum(gains[sel], 0.0)
                profits.append(float(realized.sum()) * vol)
            return _np.asarray(_np.mean(profits), dtype=_np.float32)
        return tf.numpy_function(_np_profit, [y_true, y_pred], tf.float32)
    # Nom pour l'historique Keras
    profit_metric.__name__ = 'profit'
    return profit_metric
    # Custom run_trial method to choose between 'val_loss' and 'directional_accuracy'
    
class CustomHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # Ne pas modifier dynamiquement l'objectif; utiliser celui défini à l'initialisation
        return super(CustomHyperband, self).run_trial(trial, *args, **kwargs)

class CustomRandomSearch(kt.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Ne pas modifier dynamiquement l'objectif; utiliser celui défini à l'initialisation
        return super(CustomRandomSearch, self).run_trial(trial, *args, **kwargs)

# la fermeture permet d'obtenir une fonction avec la bonne signature
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
        input_tensor = Input(shape=(data_info['look_back_x'], len(data_info['features'])))

        # Initialiser x avec le tenseur d'entrée
        x = input_tensor
        
        # Helpers pour listes/valeurs uniques
        def as_list(v):
            return v if isinstance(v, (list, tuple)) else [v]

        # Sélection de l'architecture (peut être une liste pour tuning)
        architectures = as_list(hps.get('architecture', 'lstm'))
        architecture = hp.Choice('architecture', values=architectures)

        num_layers = hp.Choice('num_layers', values=as_list(hps['layers']))
        dropout_candidates = as_list(hps.get('transformer_dropout', 0.0))
        dropout_rate = hp.Choice('dropout', values=[float(x) for x in dropout_candidates]) if len(dropout_candidates) > 1 else float(dropout_candidates[0])
        for i in range(num_layers):
            units = hp.Choice(f'units_{i}', values=as_list(hps['nb_units']))
            if architecture == 'gru':
                x = GRU(units, return_sequences=(i != num_layers - 1), dropout=dropout_rate)(x)
            elif architecture == 'transformer':
                # Simplified Transformer encoder block
                heads_candidates = as_list(hps.get('transformer_num_heads', [2, 4]))
                num_heads = hp.Choice('num_heads', values=[int(h) for h in heads_candidates]) if len(heads_candidates) > 1 else int(heads_candidates[0])
                attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=max(8, units // 4))(x, x)
                attn_output = Dropout(dropout_rate)(attn_output)
                x = x + attn_output
                x = LayerNormalization(epsilon=1e-6)(x)
                ffm_candidates = as_list(hps.get('transformer_ff_multiplier', 4))
                ff_factor = hp.Choice('ff_multiplier', values=[int(v) for v in ffm_candidates]) if len(ffm_candidates) > 1 else int(ffm_candidates[0])
                ffn = Dense(units * ff_factor, activation='relu')(x)
                ffn = Dropout(dropout_rate)(ffn)
                ffn = Dense(units)(ffn)
                x = x + ffn
                x = LayerNormalization(epsilon=1e-6)(x)
                if i != num_layers - 1:
                    # garder la séquence
                    pass
                else:
                    # réduire la séquence via pooling simple: prendre le dernier pas
                    x = Lambda(lambda t: t[:, -1, :])(x)
            else:
                x = LSTM(units, return_sequences=(i != num_layers - 1), dropout=dropout_rate)(x)

        # Si transformer avec return_sequences True encore, réduire à la fin
        if len(x.shape) == 3:
            x = Lambda(lambda t: t[:, -1, :])(x)

        # Couche de sortie
        main_output = Dense(data_info['nb_y'], name='main_output')(x)

        # Créer le modèle avec une seule sortie
        model = Model(inputs=input_tensor, outputs=main_output)

        # Compiler le modèle
        optimizer = Adam(hp.Choice('learning_rate', values=as_list(hps['learning_rate'])))
        return_type = data_info.get('return_type', 'value')

        # Normaliser la fonction de perte si nécessaire
        loss_choices = as_list(hps['loss'])
        loss_name = hp.Choice('loss', values=loss_choices) if len(loss_choices) > 1 else loss_choices[0]
        loss_fn = tf.keras.losses.Huber() if isinstance(loss_name, str) and loss_name == 'huber_loss' else loss_name

        if return_type == 'yield':
            metrics = [directional_accuracy]
            k_trades = int(data_info.get('k_trades') or 0)
            trade_volume = float(data_info.get('trade_volume') or 0)
            if k_trades and trade_volume:
                metrics.append(make_profit_metric(k_trades, trade_volume))
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        else:
            model.compile(optimizer=optimizer, loss=loss_fn)

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
        
        # Récupération de directional_accuracy (fallback pour sorties nommées)
        da_values = trial.metrics.get_history('directional_accuracy') or trial.metrics.get_history('main_output_directional_accuracy')
        val_da_values = trial.metrics.get_history('val_directional_accuracy') or trial.metrics.get_history('val_main_output_directional_accuracy')
        
        # Calcul de la moyenne de directional_accuracy sur les époques
        avg_da = np.mean([m.value for m in da_values]) if da_values else None
        avg_val_da = np.mean([m.value for m in val_da_values]) if val_da_values else None
        
        # Récupération de la perte
        loss_values = trial.metrics.get_history('loss')
        val_loss_values = trial.metrics.get_history('val_loss')
        
        # Calcul de la moyenne de la perte sur les époques
        avg_loss = np.mean([m.value for m in loss_values]) if loss_values else None
        avg_val_loss = np.mean([m.value for m in val_loss_values]) if val_loss_values else None
        
        # Enregistrement des résultats
        results.append({
            'trial_id': trial_id,
            'hyperparameters': trial.hyperparameters.values,
            'avg_directional_accuracy': avg_da,
            'avg_val_directional_accuracy': avg_val_da,
            'avg_loss': avg_loss,
            'avg_val_loss': avg_val_loss,
            'score': trial.score,
            'status': trial.status
        })
        
    return pd.DataFrame(results)

def train_and_select_best_model(data_info, hps, trainX, trainY, testX, testY, callbacks=None):
    build_model_func = build_model_closure(data_info, hps)
    
    # Choix de l'objectif en fonction du type de retour
    is_yield = data_info.get('return_type', 'value') == 'yield'
    # Objectif: maximiser le profit si disponible, sinon DA si 'yield', sinon min loss
    has_profit = bool((data_info.get('k_trades') or 0) and (data_info.get('trade_volume') or 0))
    objective_name = "val_profit" if has_profit else ("val_directional_accuracy" if is_yield else "val_loss")
    objective_dir = "max" if (has_profit or is_yield) else "min"
    method = hps.get('tuning_method', 'random')
    if method == 'hyperband':
        tuner = CustomHyperband(
            build_model_func,
            objective=kt.Objective(objective_name, direction=objective_dir),
            max_epochs=hps.get('max_epochs', 20),
            factor=hps.get('factor', 3),
            hyperband_iterations=hps.get('hyperband_iterations', 1),
            directory=hps['directory'],
            project_name=hps['project_name']
        )
    elif method == 'bayesian':
        try:
            tuner = kt.BayesianOptimization(
                build_model_func,
                objective=kt.Objective(objective_name, direction=objective_dir),
                max_trials=hps['max_trials'],
                directory=hps['directory'],
                project_name=hps['project_name']
            )
        except Exception:
            tuner = CustomRandomSearch(
                build_model_func,
                objective=kt.Objective(objective_name, direction=objective_dir),
                max_trials=hps['max_trials'],
                executions_per_trial=hps['executions_per_trial'],
                directory=hps['directory'],
                project_name=hps['project_name']
            )
    elif method == 'grid':
        # Simuler un grid search en fixant explicitement chaque combinaison via RandomSearch, avec max_trials = produit des tailles
        tuner = CustomRandomSearch(
            build_model_func,
            objective=kt.Objective(objective_name, direction=objective_dir),
            max_trials=hps['max_trials'],
            executions_per_trial=hps['executions_per_trial'],
            directory=hps['directory'],
            project_name=hps['project_name']
        )
        # Note: KerasTuner n'a pas de GridSearch natif stable, on simule via espace discret
    else:
        tuner = CustomRandomSearch(
            build_model_func,
            objective=kt.Objective(objective_name, direction=objective_dir),
            max_trials=hps['max_trials'],
            executions_per_trial=hps['executions_per_trial'],
            directory=hps['directory'],
            project_name=hps['project_name']
        )
    
    monitor_metric = objective_name
    search_callbacks = [EarlyStopping(monitor=monitor_metric, patience=hps['patience'], mode='max' if is_yield else 'min')]
    if callbacks:
        search_callbacks.extend(callbacks)
    
    tuner.search(trainX, trainY, epochs=hps['epochs'], validation_data=(testX, testY), callbacks=search_callbacks)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    tuner.trainX = trainX
    tuner.trainY = trainY
    tuner.testX = testX
    tuner.testY = testY
    
    return best_model, best_hps, tuner

def train_with_hyperband(data_info, hps, trainX, trainY, testX, testY):
    build_model_func = build_model_closure(data_info, hps)

    is_yield = data_info.get('return_type', 'value') == 'yield'
    objective_name = "val_directional_accuracy" if is_yield else "val_loss"
    objective_dir = "max" if is_yield else "min"
    tuner = CustomHyperband(
        build_model_func,
        objective=kt.Objective(objective_name, direction=objective_dir),
        max_epochs=hps['max_epochs'],
        factor=hps['factor'],
        hyperband_iterations=hps['hyperband_iterations'],
        directory=hps['directory'],
        project_name=hps['project_name']
    )

    early_stopping = EarlyStopping(monitor=objective_name, patience=hps['patience'], mode='max' if is_yield else 'min')

    # Créer les données de validation
    def create_validation_data(trainX, trainY, val_split=0.2):
        split_point = int(len(trainX) * (1 - val_split))
        valX = trainX[split_point:]
        valY = trainY[split_point:]
        trainX_new = trainX[:split_point]
        trainY_new = trainY[:split_point]
        return trainX_new, trainY_new, valX, valY

    trainX, trainY, valX, valY = create_validation_data(trainX, trainY)
    
    tuner.search(trainX, trainY, epochs=hps['epochs'], validation_data=(valX, valY), callbacks=[early_stopping])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Récupérer l'historique de l'entraînement
    history = model.fit(trainX, trainY, epochs=hps['epochs'], validation_data=(testX, testY))

    # Faire des prédictions
    predictions = model.predict(testX)

    return model, history, predictions, tuner
