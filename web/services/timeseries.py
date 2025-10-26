import pandas as pd
import numpy as np
import logging
import time


def fetch_intraday_series(shM, symbols, start_dt, end_dt_exclusive):
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(symbols), symbols, op='|')
    try:
        for share in dfShares.itertuples():
            shM.updateShareCotations(share, checkDuplicate=False)
    except Exception as upd_err:
        logging.warning(f"Mise à jour des cotations échouée (fetch_intraday_series): {upd_err}")
    listDf = shM.getListDfDataFromDfShares(dfShares, start_dt, end_dt_exclusive)
    series_map = {}
    for i, df in enumerate(listDf):
        if df is None or df.empty:
            continue
        sym = symbols[i] if i < len(symbols) else f's{i}'
        s = pd.to_numeric(df.get('openPrice', pd.Series(dtype=float)), errors='coerce')
        ts = pd.Series(s.values, index=pd.to_datetime(df.index)).sort_index().dropna()
        if ts.index.has_duplicates:
            ts = ts[~ts.index.duplicated(keep='last')]
        series_map[sym] = ts
    return series_map


def fetch_intraday_dataframe(shM, symbol, start_dt, end_dt_exclusive):
    """Retourne un DataFrame temporel (openPrice, volume, dividend) pour un symbole.
    Index datetime, trié, sans doublons, NaN gérés.
    """
    dfShares = shM.getRowsDfByKeysValues(['symbol'], [symbol])
    if dfShares.empty:
        return pd.DataFrame(columns=['openPrice', 'volume', 'dividend'])
    share = dfShares.iloc[0]
    df = shM.getDfDataRangeFromShare(share, start_dt, end_dt_exclusive)
    if df is None or df.empty:
        return pd.DataFrame(columns=['openPrice', 'volume', 'dividend'])
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep='last')]
    # Harmoniser les colonnes attendues
    if 'Open' in df.columns and 'openPrice' not in df.columns:
        df['openPrice'] = pd.to_numeric(df['Open'], errors='coerce')
    if 'Volume' in df.columns and 'volume' not in df.columns:
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    if 'Dividends' in df.columns and 'dividend' not in df.columns:
        df['dividend'] = pd.to_numeric(df['Dividends'], errors='coerce')
    cols = [c for c in ['openPrice', 'volume', 'dividend'] if c in df.columns]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna(how='all')
    return df


def fetch_intraday_series_with_perf(shM, symbols, start_dt, end_dt_exclusive):
    t0 = time.perf_counter()
    t_db0 = time.perf_counter()
    dfShares = shM.getRowsDfByKeysValues(['symbol'] * len(symbols), symbols, op='|')
    t_db1 = time.perf_counter()
    t_upd0 = time.perf_counter()
    upd_errors = 0
    try:
        for share in dfShares.itertuples():
            try:
                shM.updateShareCotations(share, checkDuplicate=False)
            except Exception:
                upd_errors += 1
    except Exception as upd_err:
        logging.warning(f"Mise à jour cotations (fetch_intraday_series_with_perf) erreur: {upd_err}")
    t_upd1 = time.perf_counter()
    t_list0 = time.perf_counter()
    listDf = shM.getListDfDataFromDfShares(dfShares, start_dt, end_dt_exclusive)
    t_list1 = time.perf_counter()
    t_build0 = time.perf_counter()
    series_map = {}
    num_empty = 0
    total_points = 0
    for i, df in enumerate(listDf):
        if df is None or df.empty:
            num_empty += 1
            continue
        sym = symbols[i] if i < len(symbols) else f's{i}'
        s = pd.to_numeric(df.get('openPrice', pd.Series(dtype=float)), errors='coerce')
        ts = pd.Series(s.values, index=pd.to_datetime(df.index)).sort_index().dropna()
        if ts.index.has_duplicates:
            ts = ts[~ts.index.duplicated(keep='last')]
        series_map[sym] = ts
        total_points += len(ts)
    t_build1 = time.perf_counter()
    perf = {
        'total_s': float(t_build1 - t0),
        'db_query_s': float(t_db1 - t_db0),
        'update_cotations_s': float(t_upd1 - t_upd0),
        'get_list_s': float(t_list1 - t_list0),
        'build_series_s': float(t_build1 - t_build0),
        'symbols': int(len(symbols)),
        'series_built': int(len(series_map)),
        'series_empty': int(num_empty),
        'total_points': int(total_points),
        'update_errors': int(upd_errors),
    }
    return series_map, perf


def align_minute(series_map, start_dt, end_dt_exclusive):
    if not series_map:
        return pd.DataFrame()
    combined = pd.concat(series_map, axis=1).sort_index()
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep='last')]
    resampled_min = combined.resample('1min').last().ffill().bfill()
    aligned = resampled_min[(resampled_min.index >= start_dt) & (resampled_min.index < end_dt_exclusive)].dropna(how='any')
    return aligned


def align_minute_with_perf(series_map, start_dt, end_dt_exclusive):
    t0 = time.perf_counter()
    if not series_map:
        return pd.DataFrame(), {'total_s': 0.0}
    t_cat0 = time.perf_counter()
    combined = pd.concat(series_map, axis=1).sort_index()
    if combined.index.has_duplicates:
        combined = combined[~combined.index.duplicated(keep='last')]
    t_cat1 = time.perf_counter()
    t_res0 = time.perf_counter()
    # Resample, forward-fill, backward-fill. Conserve les lignes où au moins une série a des données.
    resampled_min = combined.resample('1min').last().ffill().bfill()
    t_res1 = time.perf_counter() # <--- Correction ici, t_res1 doit être défini avant t_win0
    t_win0 = time.perf_counter()
    # Ensuite, on garde la fenêtre et on supprime les lignes totalement vides
    aligned = resampled_min[(resampled_min.index >= start_dt) & (resampled_min.index < end_dt_exclusive)].dropna(how='all')
    t_win1 = time.perf_counter()
    perf = {
        'total_s': float(t_win1 - t0),
        'concat_dedup_s': float(t_cat1 - t_cat0),
        'resample_ffill_bfill_s': float(t_res1 - t_res0),
        'window_dropna_s': float(t_win1 - t_win0),
        'rows_out': int(aligned.shape[0]),
        'cols': int(aligned.shape[1]) if not aligned.empty else 0,
    }
    return aligned, perf


def daily_cutoff_mask(aligned_index, minutes_before_last=10):
    norm_index = aligned_index.normalize()
    day_last_idx = aligned_index.to_series().groupby(norm_index).max()
    cutoff_by_day = day_last_idx - pd.Timedelta(minutes=minutes_before_last)
    cutoff_for_rows = norm_index.map(cutoff_by_day)
    return aligned_index >= cutoff_for_rows.values


