"""
Fonctions de préparation des données pour l'entraînement et l'inférence.
"""

import pandas as pd
import numpy as np
import logging
from io import StringIO
from typing import Tuple, List, Optional


def estimate_nb_quotes_per_day(df: pd.DataFrame) -> int:
    """
    Estime le nombre de quotes par jour à partir d'un DataFrame.
    
    Args:
        df: DataFrame avec un index DateTimeIndex
    
    Returns:
        Nombre estimé de quotes par jour
    """
    if df is None or df.empty:
        return 0
    
    try:
        days = df.index.normalize().unique()
        if len(days) == 0:
            return 0
        
        counts = [len(df[df.index.normalize() == d]) for d in days]
        return int(np.mean(counts)) if counts else 0
    except Exception:
        return len(df) // max(1, len(df.index.normalize().unique()))


def prepare_xy_from_store(
    store_json: str,
    look_back: int,
    stride: int,
    nb_y: int,
    first_minutes: Optional[int] = None,
    prediction_type: str = 'return'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Prépare les batches X et Y pour l'entraînement.
    
    Args:
        store_json: JSON contenant les données (format pandas split)
        look_back: Taille de la fenêtre d'entrée
        stride: Pas d'échantillonnage
        nb_y: Nombre de points futurs à prédire
        first_minutes: Nombre de minutes d'observation (ou look_back * stride si None)
        prediction_type: 'return' (variation relative), 'price' (prix normalisé), ou 'signal'
    
    Returns:
        Tuple (trainX, trainY, testX, testY, nb_per_day)
    """
    if not store_json:
        return None, None, None, None, 0
    
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
    except Exception as e:
        logging.error(f"[DataPrep] Erreur parsing JSON: {e}")
        return None, None, None, None, 0
    
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

    def create_xy(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        if dataset is None or dataset.empty:
            return np.zeros((0, look_back, 1), dtype=float), np.zeros((0, nb_y), dtype=float)
        
        norm = dataset.index.normalize()
        days_u = norm.unique()
        
        for d in days_u:
            day_df = dataset.loc[norm == d, ['openPrice']]
            if day_df.shape[0] < obs_window + max(2, nb_y):
                continue
            
            # Sélectionner les points pour construire la fenêtre d'entrée
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
            if remainder <= 0 or remainder <= nb_y:
                continue
            
            stride_y = remainder // (nb_y + 1)
            if stride_y == 0:
                continue
            
            offsets = [(j + 1) * stride_y for j in range(nb_y)]
            
            y_vals = []
            prev_price = base_price
            prices_list = [base_price]
            
            if prediction_type == 'signal':
                horizon = offsets[-1] if offsets else 1
                if obs_window + horizon < day_df.shape[0]:
                    final_p = float(day_df.iloc[obs_window + horizon, 0])
                    ret = (final_p - base_price) / base_price
                    
                    if ret < -0.005: label = 0
                    elif ret < -0.001: label = 1
                    elif ret < 0.001: label = 2
                    elif ret < 0.005: label = 3
                    else: label = 4
                    
                    y_vals.append(label)
                    prices_list.append(final_p)
            else:
                for i, off in enumerate(offsets):
                    y_price = float(day_df.iloc[obs_window + off, 0])
                    prices_list.append(y_price)
                    
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
    first_minutes: Optional[int] = None,
    prediction_type: str = 'return'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], int, List]:
    """
    Prépare X/Y pour l'inférence (généralisation) sur la courbe courante.
    
    Args:
        store_json: JSON contenant les données
        look_back: Taille de la fenêtre d'entrée
        stride: Pas d'échantillonnage
        nb_y: Nombre de points futurs à prédire
        first_minutes: Nombre de minutes d'observation
        prediction_type: Type de prédiction
    
    Returns:
        Tuple (X, Y, df, obs_window, sample_days)
    """
    if not store_json:
        return None, None, None, 0, []
    
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
    except Exception:
        return None, None, None, 0, []
    
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

