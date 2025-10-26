import pandas as pd
import numpy as np
import logging
from datetime import time as _time
from .timeseries import fetch_intraday_series


MAIN_HOURS_START = _time(9, 30)
MAIN_HOURS_END = _time(16, 0)


def _filter_between_times(df: pd.DataFrame, start_t: _time, end_t: _time) -> pd.DataFrame:
    """
    @brief Filtre un DataFrame/Series temporel entre deux heures
    
    @param df DataFrame ou Series avec index temporel
    @param start_t Heure de début (incluse)
    @param end_t Heure de fin (exclue)
    @return DataFrame filtré sur la plage horaire
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    start_str = start_t.strftime('%H:%M')
    end_str = end_t.strftime('%H:%M')
    try:
        return df.between_time(start_str, end_str, inclusive='left')
    except TypeError:
        idx = df.index
        mask = (idx.time >= start_t) & (idx.time < end_t)
        return df[mask]


def _get_hours_map(shM, symbols):
    """
    @brief Retourne un dictionnaire des heures d'ouverture/fermeture par symbole
    
    @param shM Objet SharesManager contenant les données des actions
    @param symbols Liste des symboles à traiter
    @return Dict {symbol: (open_time, close_time)} avec fallback sur MAIN_HOURS_*
    """
    hours = {}
    try:
        df = shM.getRowsDfByKeysValues(['symbol'] * len(symbols), symbols, op='|')
    except Exception:
        df = None
    for s in symbols:
        open_t = MAIN_HOURS_START
        close_t = MAIN_HOURS_END
        try:
            if df is not None and not df.empty:
                row = df[df['symbol'] == s]
                if not row.empty and 'openMarketTime' in row.columns and 'closeMarketTime' in row.columns:
                    ot = row.iloc[0].openMarketTime
                    ct = row.iloc[0].closeMarketTime
                    if ot is not None:
                        open_t = ot
                    if ct is not None:
                        close_t = ct
        except Exception:
            pass
        hours[s] = (open_t, close_t)
    return hours


def _cluster_symbols_by_hours(hours_map, bucket_minutes: int = 30):
    """
    @brief Regroupe les symboles par fenêtres horaires
    
    @param hours_map Dictionnaire des heures par symbole
    @param bucket_minutes Taille des buckets en minutes (défaut: 30)
    @return Liste de listes de symboles triée par taille décroissante
    """
    def bucketize(t: _time) -> int:
        return int(round((t.hour * 60 + t.minute) / float(bucket_minutes))) * bucket_minutes
    clusters = {}
    for s, (ot, ct) in hours_map.items():
        key = (bucketize(ot), bucketize(ct))
        clusters.setdefault(key, []).append(s)
    return sorted(clusters.values(), key=lambda xs: len(xs), reverse=True)


def _align_minute_from_map(shM, series_map, start_dt, end_dt_exclusive):
    """
    @brief Aligne les séries temporelles à la minute et filtre sur les heures de marché
    
    @param shM Objet SharesManager
    @param series_map Dictionnaire {symbol: Series} des données
    @param start_dt Date de début
    @param end_dt_exclusive Date de fin (exclue)
    @return DataFrame aligné à la minute avec données communes
    """
    if not series_map:
        return pd.DataFrame()
    symbols = list(series_map.keys())
    hours_map = _get_hours_map(shM, symbols)
    # Resample + filtre main hours par symbole
    resampled_filtered = {}
    for s, ser in series_map.items():
        ser = ser.sort_index()
        if ser.index.has_duplicates:
            ser = ser[~ser.index.duplicated(keep='last')]
        ser1m = ser.resample('1T').last().ffill().bfill()
        ser1m = ser1m[(ser1m.index >= start_dt) & (ser1m.index < end_dt_exclusive)]
        ot, ct = hours_map.get(s, (MAIN_HOURS_START, MAIN_HOURS_END))
        ser1m = _filter_between_times(ser1m, ot, ct)
        resampled_filtered[s] = ser1m
    combined = pd.concat(resampled_filtered, axis=1).sort_index()
    # Garder uniquement les instants communs à toutes les colonnes présentes (évite chevauchements faibles)
    aligned = combined.dropna(how='any')
    return aligned


def get_minute_returns(shM, symbols, start_dt, end_dt_exclusive):
    """
    @brief Calcule les rendements minute pour une liste de symboles
    
    @param shM Objet SharesManager
    @param symbols Liste des symboles
    @param start_dt Date de début
    @param end_dt_exclusive Date de fin (exclue)
    @return DataFrame des rendements minute (pct_change)
    """
    # Exclure paires avec heures de marché disjointes (ex: US vs EU) en limitant aux symboles
    # partageant des hours similaires; sinon l'intersection tombera à quasi rien.
    series_map = fetch_intraday_series(shM, symbols, start_dt, end_dt_exclusive)
    if len(series_map) < 2:
        return pd.DataFrame()
    aligned = _align_minute_from_map(shM, series_map, start_dt, end_dt_exclusive)
    if aligned.shape[0] == 0:
        return pd.DataFrame()
    returns = aligned.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    # Retirer colonnes constantes
    valid_cols = [c for c in returns.columns if returns[c].std(skipna=True) and returns[c].std(skipna=True) > 0]
    return returns[valid_cols] if len(valid_cols) >= 2 else pd.DataFrame()


def get_returns_with_daily_fallback(shM, symbols, start_dt, end_dt_exclusive):
    """
    @brief Calcule les rendements avec fallback quotidien si données minute insuffisantes
    
    @param shM Objet SharesManager
    @param symbols Liste des symboles
    @param start_dt Date de début
    @param end_dt_exclusive Date de fin (exclue)
    @return DataFrame des rendements (minute ou quotidien selon disponibilité)
    """
    series_map = fetch_intraday_series(shM, symbols, start_dt, end_dt_exclusive)
    if len(series_map) < 2:
        return pd.DataFrame()
    # Minute (main hours par symbole)
    aligned_min = _align_minute_from_map(shM, series_map, start_dt, end_dt_exclusive)
    if aligned_min.shape[0] >= 10:
        returns = aligned_min.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
        valid_cols = [c for c in returns.columns if returns[c].std(skipna=True) and returns[c].std(skipna=True) > 0]
        if len(valid_cols) >= 2:
            return returns[valid_cols]
    # Fallback day
    combined = pd.concat(series_map, axis=1).sort_index()
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep='last')]
    resampled_day = combined.resample('1D').last().ffill().bfill()
    aligned_day = resampled_day[(resampled_day.index >= start_dt) & (resampled_day.index < end_dt_exclusive)].dropna(how='any')
    if aligned_day.shape[0] < 5:
        return pd.DataFrame()
    returns = aligned_day.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    valid_cols = [c for c in returns.columns if returns[c].std(skipna=True) and returns[c].std(skipna=True) > 0]
    return returns[valid_cols] if len(valid_cols) >= 2 else pd.DataFrame()


def _compute_pair_corr(left: pd.Series,
                       right: pd.Series,
                       positive_mode: str = 'none',
                       min_points: int = 60,
                       min_fraction: float = 0.1,
                       winsor_limit: float = 0.01,
                       method: str = 'spearman') -> float:
    """
    @brief Calcule la corrélation entre deux séries avec filtres et winsorisation
    
    @param left Première série
    @param right Deuxième série
    @param positive_mode Mode de filtrage: 'none', 'both_positive', 'left_positive'
    @param min_points Nombre minimum de points requis
    @param min_fraction Fraction minimum de données requise
    @param winsor_limit Limite de winsorisation (0.01 = 1%)
    @param method Méthode de corrélation: 'spearman' ou 'pearson'
    @return Coefficient de corrélation ou NaN si insuffisant
    """
    common = pd.concat([left, right], axis=1, join='inner').dropna()
    if common.empty:
        return np.nan
    total_len = common.shape[0]
    df = common
    if positive_mode == 'both_positive':
        df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]
    elif positive_mode == 'left_positive':
        df = df[(df.iloc[:, 0] > 0)]
    if df.shape[0] < min_points or df.shape[0] < int(np.floor(min_fraction * total_len)):
        return np.nan

    # Winsorisation légère pour éviter l'influence des extrêmes
    if winsor_limit and winsor_limit > 0:
        ql0, qh0 = df.iloc[:, 0].quantile(winsor_limit), df.iloc[:, 0].quantile(1 - winsor_limit)
        ql1, qh1 = df.iloc[:, 1].quantile(winsor_limit), df.iloc[:, 1].quantile(1 - winsor_limit)
        s0 = df.iloc[:, 0].clip(ql0, qh0)
        s1 = df.iloc[:, 1].clip(ql1, qh1)
    else:
        s0 = df.iloc[:, 0]
        s1 = df.iloc[:, 1]

    try:
        return s0.corr(s1, method=method)
    except Exception:
        return np.nan


def corr_matrix_with_lag(returns: pd.DataFrame, lag_minutes: int, positive_mode: str = 'none') -> pd.DataFrame:
    """
    @brief Calcule la matrice de corrélation avec décalage temporel
    
    @param returns DataFrame des rendements
    @param lag_minutes Décalage en minutes (0 = pas de décalage)
    @param positive_mode Mode de filtrage sur les hausses
    @return Matrice de corrélation
    """
    if returns.empty:
        return pd.DataFrame()
    cols = [c for c in returns.columns]
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in cols:
        base_i = returns[i]
        for j in cols:
            if lag_minutes == 0:
                right = returns[j]
            else:
                right = returns[j].shift(int(lag_minutes))
            mat.loc[i, j] = _compute_pair_corr(base_i, right, positive_mode=positive_mode)
    return mat


def corr_matrix_with_lag_fast(returns: pd.DataFrame, lag_minutes: int, positive_mode: str = 'none', method: str = 'pearson') -> pd.DataFrame:
    """
    Version vectorisée (NumPy) pour corrélation Pearson par paires sous un lag fixe.
    - Suppose des séries centrées/normalisées par paire après jointure des index.
    - positive_mode géré par masque après jointure.
    """
    if returns.empty:
        return pd.DataFrame()
    cols = list(returns.columns)
    n = len(cols)
    out = np.full((n, n), np.nan, dtype=float)
    for a in range(n):
        sA = returns.iloc[:, a]
        for b in range(n):
            sB = returns.iloc[:, b]
            if lag_minutes:
                sB = sB.shift(int(lag_minutes))
            df = pd.concat([sA, sB], axis=1, join='inner').dropna()
            if df.shape[0] < 5:
                continue
            if positive_mode == 'both_positive':
                df = df[(df.iloc[:, 0] > 0) & (df.iloc[:, 1] > 0)]
            elif positive_mode == 'left_positive':
                df = df[(df.iloc[:, 0] > 0)]
            if df.shape[0] < 5:
                continue
            x = df.iloc[:, 0].to_numpy(dtype=float, copy=False)
            y = df.iloc[:, 1].to_numpy(dtype=float, copy=False)
            # Pearson rapide via produits scalaires
            x = x - x.mean()
            y = y - y.mean()
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            if denom <= 0:
                continue
            out[a, b] = float(np.dot(x, y) / denom)
    return pd.DataFrame(out, index=cols, columns=cols)


def corr_matrix_max_0_30(returns: pd.DataFrame, positive_mode: str = 'none'):
    """
    @brief Trouve la corrélation maximale sur les décalages 0-30 minutes
    
    @param returns DataFrame des rendements
    @param positive_mode Mode de filtrage sur les hausses
    @return Tuple (matrice_corr_max, matrice_best_lag)
    """
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()
    cols = [c for c in returns.columns]
    max_mat = pd.DataFrame(-np.inf, index=cols, columns=cols, dtype=float)
    best_lag = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for k in range(0, 31):
        cur = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i in cols:
            base_i = returns[i]
            for j in cols:
                right = returns[j] if k == 0 else returns[j].shift(k)
                cur.loc[i, j] = _compute_pair_corr(base_i, right, positive_mode=positive_mode)
        cur = cur.fillna(-np.inf)
        better = cur > max_mat
        max_mat = max_mat.where(~better, cur)
        best_lag = best_lag.where(~better, k)
    if not np.isfinite(max_mat.values).any():
        return pd.DataFrame(), pd.DataFrame()
    return max_mat.replace(-np.inf, np.nan), best_lag


def daily_crosscorr_points(shM, symbol1: str, symbol2: str, start_dt, end_dt_exclusive):
    """
    @brief Calcule les points de corrélation quotidienne entre deux symboles
    
    @param shM Objet SharesManager
    @param symbol1 Premier symbole
    @param symbol2 Deuxième symbole
    @param start_dt Date de début
    @param end_dt_exclusive Date de fin (exclue)
    @return Tuple (xs, ys, texts) pour visualisation
    """
    # Use minute returns only
    returns = get_minute_returns(shM, [symbol1, symbol2], start_dt, end_dt_exclusive)
    if returns.empty or symbol1 not in returns.columns or symbol2 not in returns.columns:
        return [], [], []
    returns = returns.copy()
    returns['date'] = returns.index.date
    groups = returns.groupby('date')
    xs, ys, texts = [], [], []
    for day, df_day in groups:
        if df_day.shape[0] < 5:
            continue
        base1 = df_day[symbol1]
        base2 = df_day[symbol2]
        for k in range(0, 31):
            if k == 0:
                corr = base1.corr(base2)
            else:
                corr = base1.corr(base2.shift(k))
            if pd.isna(corr):
                continue
            xs.append(pd.to_datetime(day))
            ys.append(corr)
            texts.append(f"lag={k} min")
    return xs, ys, texts



# --- Nouvelles fonctions: corrélation combinée (retours 1m + taux sur fenêtre) ---

def get_returns_and_roc(shM, symbols, start_dt, end_dt_exclusive, roc_window_minutes: int = 30):
    """
    @brief Calcule les rendements minute et les taux de variation (ROC)
    
    @param shM Objet SharesManager
    @param symbols Liste des symboles
    @param start_dt Date de début
    @param end_dt_exclusive Date de fin (exclue)
    @param roc_window_minutes Fenêtre pour le calcul ROC (défaut: 30 min)
    @return Tuple (returns_df, roc_df) avec mêmes colonnes
    """
    series_map = fetch_intraday_series(shM, symbols, start_dt, end_dt_exclusive)
    if len(series_map) < 2:
        return pd.DataFrame(), pd.DataFrame()
    aligned = _align_minute_from_map(series_map, start_dt, end_dt_exclusive)
    if aligned.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()
    returns = aligned.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how='any')
    # Colonnes valides: variance non nulle
    valid_cols = [c for c in returns.columns if returns[c].std(skipna=True) and returns[c].std(skipna=True) > 0]
    if len(valid_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()
    returns = returns[valid_cols]
    roc = aligned.pct_change(periods=int(max(1, roc_window_minutes))).replace([np.inf, -np.inf], np.nan)
    # Restreindre aux mêmes colonnes et index des returns pour éviter des NaN initiaux
    roc = roc[valid_cols].reindex(returns.index)
    # On ne drop pas toutes les lignes NA ici; la corrélation gère des indices communs par jointure
    return returns, roc


def _combine_two_corr_matrices(mat_returns: pd.DataFrame, mat_roc: pd.DataFrame, w_returns: float, w_roc: float) -> pd.DataFrame:
    """
    @brief Combine deux matrices de corrélation avec des poids
    
    @param mat_returns Matrice de corrélation des rendements
    @param mat_roc Matrice de corrélation des taux de variation
    @param w_returns Poids pour les rendements
    @param w_roc Poids pour les taux de variation
    @return Matrice combinée avec moyenne pondérée
    """
    if mat_returns is None or mat_returns.empty:
        return mat_roc
    if mat_roc is None or mat_roc.empty:
        return mat_returns
    # Assurer même index/colonnes
    idx = mat_returns.index
    cols = mat_returns.columns
    mat_roc = mat_roc.reindex(index=idx, columns=cols)
    combined = mat_returns.copy().astype(float)
    both_mask = mat_returns.notna() & mat_roc.notna()
    only_r_mask = mat_returns.notna() & ~mat_roc.notna()
    only_k_mask = ~mat_returns.notna() & mat_roc.notna()
    # Remplir valeurs
    combined[only_r_mask] = mat_returns[only_r_mask]
    combined[only_k_mask] = mat_roc[only_k_mask]
    combined[both_mask] = (
        w_returns * mat_returns[both_mask] + w_roc * mat_roc[both_mask]
    ) / float(max(1e-12, (w_returns + w_roc)))
    return combined


def corr_matrix_with_lag_combined(returns: pd.DataFrame,
                                  roc: pd.DataFrame,
                                  lag_minutes: int,
                                  weight_returns: float = 0.5,
                                  weight_roc: float = 0.5,
                                  positive_mode: str = 'none') -> pd.DataFrame:
    """
    @brief Calcule la corrélation combinée avec décalage temporel
    
    @param returns DataFrame des rendements minute
    @param roc DataFrame des taux de variation
    @param lag_minutes Décalage en minutes
    @param weight_returns Poids pour les rendements (défaut: 0.5)
    @param weight_roc Poids pour les taux de variation (défaut: 0.5)
    @param positive_mode Mode de filtrage sur les hausses
    @return Matrice de corrélation combinée
    """
    if returns is None or returns.empty:
        return pd.DataFrame()
    # Corr retours
    mat_r = corr_matrix_with_lag(returns, int(lag_minutes), positive_mode=positive_mode)
    # Corr taux
    mat_k = pd.DataFrame()
    if roc is not None and not roc.empty:
        mat_k = corr_matrix_with_lag(roc, int(lag_minutes), positive_mode=positive_mode)
    # Combine
    return _combine_two_corr_matrices(mat_r, mat_k, weight_returns, weight_roc)


def corr_matrix_max_0_30_combined(returns: pd.DataFrame,
                                  roc: pd.DataFrame,
                                  max_lag: int = 30,
                                  weight_returns: float = 0.5,
                                  weight_roc: float = 0.5,
                                  positive_mode: str = 'none'):
    """
    @brief Trouve la corrélation combinée maximale sur plusieurs décalages
    
    @param returns DataFrame des rendements minute
    @param roc DataFrame des taux de variation
    @param max_lag Décalage maximum à tester (défaut: 30)
    @param weight_returns Poids pour les rendements (défaut: 0.5)
    @param weight_roc Poids pour les taux de variation (défaut: 0.5)
    @param positive_mode Mode de filtrage sur les hausses
    @return Tuple (matrice_corr_max, matrice_best_lag)
    """
    if returns is None or returns.empty:
        return pd.DataFrame(), pd.DataFrame()
    left_cols = [c for c in returns.columns]
    max_mat = pd.DataFrame(-np.inf, index=left_cols, columns=left_cols, dtype=float)
    best_lag = pd.DataFrame(0, index=left_cols, columns=left_cols, dtype=int)
    for k in range(0, int(max_lag) + 1):
        mat = corr_matrix_with_lag_combined(returns, roc, k, weight_returns, weight_roc, positive_mode=positive_mode)
        if mat is None or mat.empty:
            continue
        cur = mat.fillna(-np.inf)
        better = cur > max_mat
        max_mat = max_mat.where(~better, cur)
        best_lag = best_lag.where(~better, k)
    if not np.isfinite(max_mat.values).any():
        return pd.DataFrame(), pd.DataFrame()
    return max_mat.replace(-np.inf, np.nan), best_lag

