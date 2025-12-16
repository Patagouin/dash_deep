# -*- coding: utf-8 -*-
"""
Fonctions de préparation des données pour l'entraînement ML.
Extraites de playground.py pour réutilisation dans d'autres pages.
"""

import numpy as np
import pandas as pd
from io import StringIO
import logging

from web.services.synthetic import estimate_nb_quotes_per_day


def prepare_xy_from_store(
    store_json: str,
    look_back: int,
    stride: int,
    nb_y: int,
    first_minutes: int = None,
    prediction_type: str = 'return'
):
    """
    Prépare les batches X et Y pour l'entraînement.
    
    Args:
        store_json: Données JSON (format split)
        look_back: Taille de la fenêtre d'entrée
        stride: Pas d'échantillonnage
        nb_y: Nombre de points futurs à prédire
        first_minutes: Période d'observation (si None, utilise look_back * stride)
        prediction_type: 'return' (variation relative), 'price' (prix normalisé), ou 'signal' (classification)
    
    Returns:
        trainX, trainY, testX, testY, nb_per_day
    """
    if not store_json:
        return None, None, None, None, 0
    
    df = pd.read_json(StringIO(store_json), orient='split')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    nb_per_day = estimate_nb_quotes_per_day(df)
    
    if nb_per_day <= 0:
        return None, None, None, None, 0
    
    # Split train/test par jours (80/20)
    days = df.index.normalize().unique()
    if len(days) < 2:
        split = len(df)
        train_df = df.iloc[:split]
        test_df = df.iloc[0:0]
    else:
        split_idx = int(len(days) * 0.8)
        split_day = days[split_idx - 1]
        train_df = df.loc[df.index.normalize() <= split_day]
        test_df = df.loc[df.index.normalize() > split_day]

    obs_window = int(first_minutes) if first_minutes is not None and first_minutes > 0 else int(look_back * stride)

    def create_xy(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        if dataset is None or dataset.empty:
            return np.zeros((0, look_back, 1), dtype=float), np.zeros((0, nb_y), dtype=float)
        
        # Itérer par jour
        norm = dataset.index.normalize()
        days_u = norm.unique()
        
        for d in days_u:
            day_df = dataset.loc[norm == d, ['openPrice']]
            if day_df.shape[0] < obs_window + max(2, nb_y):
                continue
            
            # Sélectionner les obs_window premières minutes pour construire la fenêtre d'entrée
            if obs_window < look_back * stride:
                available_points = min(obs_window, day_df.shape[0])
                if available_points < look_back:
                    continue
                step = max(1, available_points // look_back)
                seq = day_df.iloc[0: available_points: step].iloc[:look_back].to_numpy(dtype=float)
            else:
                step = max(1, obs_window // look_back)
                seq = day_df.iloc[0: obs_window: step].iloc[:look_back].to_numpy(dtype=float)
            
            if seq.shape[0] != look_back:
                continue
            
            base_price = seq[-1, 0]
            if base_price == 0:
                continue
            
            # Normaliser input
            seq[:, 0] = seq[:, 0] / base_price
            if seq.shape[1] >= 2:
                seq[:, 1] = np.log1p(np.clip(seq[:, 1], a_min=0.0, a_max=None))
            
            remainder = day_df.shape[0] - obs_window
            if remainder <= 0:
                continue
            if remainder <= nb_y:
                continue
            
            stride_y = remainder // (nb_y + 1)
            if stride_y == 0:
                continue
            
            offsets = [(j + 1) * stride_y for j in range(nb_y)]
            
            y_vals = []
            prev_price = base_price
            prices_list = [base_price]  # Pour logging
            
            if prediction_type == 'signal':
                # Classification 5 classes : -2, -1, 0, 1, 2 (mappées 0..4)
                horizon = offsets[-1] if offsets else 1
                if obs_window + horizon < day_df.shape[0]:
                    final_p = float(day_df.iloc[obs_window + horizon, 0])
                    ret = (final_p - base_price) / base_price
                    
                    # Seuils arbitraires
                    if ret < -0.005:
                        label = 0   # -2 (Strong Drop)
                    elif ret < -0.001:
                        label = 1   # -1 (Drop)
                    elif ret < 0.001:
                        label = 2   # 0 (Flat)
                    elif ret < 0.005:
                        label = 3   # 1 (Rise)
                    else:
                        label = 4   # 2 (Strong Rise)
                    y_vals.append(label)
                    prices_list.append(final_p)
            else:
                for i, off in enumerate(offsets):
                    y_price = float(day_df.iloc[obs_window + off, 0])
                    prices_list.append(y_price)
                    
                    if prediction_type == 'price':
                        # Mode Prix : ratio par rapport au dernier prix connu (base_price)
                        val = y_price / base_price
                        y_vals.append(val)
                    else:
                        # Mode Return (défaut) : variations relatives pas à pas
                        if i == 0:
                            variation = (y_price / prev_price) - 1.0
                            y_vals.append(variation)
                        else:
                            prev_off = offsets[i - 1]
                            prev_price_iter = float(day_df.iloc[obs_window + prev_off, 0])
                            variation = (y_price / prev_price_iter) - 1.0
                            y_vals.append(variation)
                        prev_price = y_price
            
            # Log détaillé seulement pour le premier
            if len(X) == 0:
                logging.info(f"[Prepare XY] Mode={prediction_type}. Exemple premier échantillon:")
                logging.info(f"  Prix: {[f'{p:.2f}' for p in prices_list]}")
                logging.info(f"  Valeurs cibles: {[f'{v:.4f}' for v in y_vals]}")
            
            X.append(seq)
            Y.append(y_vals)
        
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        return X, Y

    trainX, trainY = create_xy(train_df)
    testX, testY = create_xy(test_df)
    return trainX, trainY, testX, testY, nb_per_day


def prepare_xy_for_inference(
    store_json: str,
    look_back: int,
    stride: int,
    nb_y: int,
    first_minutes: int = None,
    prediction_type: str = 'return'
):
    """
    Prépare X/Y pour l'inférence (généralisation) sur la courbe courante.
    Contrairement à prepare_xy_from_store, on ne fait pas de split train/test :
    chaque jour valide fournit un seul échantillon (X, Y) basé sur les premières minutes.
    
    Args:
        store_json: Données JSON (format split)
        look_back: Taille de la fenêtre d'entrée
        stride: Pas d'échantillonnage
        nb_y: Nombre de points futurs à prédire
        first_minutes: Période d'observation
        prediction_type: 'return' ou 'price'
    
    Returns:
        X, Y, df, obs_window, sample_days
    """
    if not store_json:
        return None, None, None, 0, []
    
    df = pd.read_json(StringIO(store_json), orient='split')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    
    if df is None or df.empty:
        return None, None, None, 0, []

    obs_window = int(first_minutes) if first_minutes is not None and first_minutes > 0 else int(look_back * stride)

    norm = df.index.normalize()
    days_u = norm.unique()

    X_list = []
    Y_list = []
    sample_days = []

    for d in days_u:
        day_df = df.loc[norm == d, ['openPrice']]
        if day_df.shape[0] < obs_window + max(2, nb_y):
            continue

        # Sélection des points d'observation
        if obs_window < look_back * stride:
            available_points = min(obs_window, day_df.shape[0])
            if available_points < look_back:
                continue
            step = max(1, available_points // look_back)
            seq = day_df.iloc[0: available_points: step].iloc[:look_back].to_numpy(dtype=float)
        else:
            step = max(1, obs_window // look_back)
            seq = day_df.iloc[0: obs_window: step].iloc[:look_back].to_numpy(dtype=float)

        if seq.shape[0] != look_back:
            continue

        base_price = seq[-1, 0]
        if base_price == 0:
            continue

        # Normaliser input
        seq[:, 0] = seq[:, 0] / base_price

        remainder = day_df.shape[0] - obs_window
        if remainder <= nb_y:
            continue

        stride_y = remainder // (nb_y + 1)
        if stride_y <= 0:
            continue

        offsets = [(j + 1) * stride_y for j in range(nb_y)]

        y_vals = []
        prev_price = base_price

        for i, off in enumerate(offsets):
            y_price = float(day_df.iloc[obs_window + off, 0])

            if prediction_type == 'price':
                val = y_price / base_price
                y_vals.append(val)
            else:
                if i == 0:
                    variation = (y_price / prev_price) - 1.0
                    y_vals.append(variation)
                else:
                    prev_off = offsets[i - 1]
                    prev_price_iter = float(day_df.iloc[obs_window + prev_off, 0])
                    variation = (y_price / prev_price_iter) - 1.0
                    y_vals.append(variation)
                prev_price = y_price

        X_list.append(seq)
        Y_list.append(y_vals)
        sample_days.append(d)

    if not X_list:
        return None, None, df, obs_window, []

    X = np.asarray(X_list, dtype=float)
    Y = np.asarray(Y_list, dtype=float)
    return X, Y, df, obs_window, sample_days

