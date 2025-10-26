import time
import numpy as np
import pandas as pd


def _infer_model_shapes(model):
    # input: (None, look_back, features)
    in_shape = getattr(model, 'input_shape', None)
    out_shape = getattr(model, 'output_shape', None)
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    if isinstance(out_shape, list):
        out_shape = out_shape[0]
    look_back = int(in_shape[1]) if in_shape and len(in_shape) >= 3 else 60
    num_features = int(in_shape[2]) if in_shape and len(in_shape) >= 3 else 1
    nb_y = int(out_shape[1]) if out_shape and len(out_shape) >= 2 else 2
    return look_back, num_features, nb_y


def _build_features(day_df: pd.DataFrame, num_features: int):
    # Always use openPrice, optionally volume if required
    cols = []
    if 'openPrice' in day_df.columns:
        cols.append(day_df['openPrice'].astype(float).values)
    else:
        # fallback, use first numeric column
        num_col = day_df.select_dtypes(include=[np.number]).columns
        if len(num_col) == 0:
            cols.append(np.zeros(len(day_df), dtype=float))
        else:
            cols.append(day_df[num_col[0]].astype(float).values)
    if num_features >= 2:
        if 'volume' in day_df.columns:
            vol = np.log1p(day_df['volume'].astype(float).clip(lower=0)).values
        else:
            vol = np.zeros(len(day_df), dtype=float)
        cols.append(vol)
    # If more features required by model, pad with zeros
    while len(cols) < num_features:
        cols.append(np.zeros(len(day_df), dtype=float))
    # shape: (len, num_features)
    feats = np.vstack(cols).T
    return feats


def backtest_model_intraday(day_aligned_df: pd.DataFrame,
                            model,
                            initial_cash: float,
                            per_trade_amount: float,
                            k_trades: int = 1):
    """
    Backtest simple basé modèle:
    - Pour chaque jour, une fenêtre unique au début (après look_back)
    - Le modèle prédit nb_y rendements (yield) à différents offsets répartis sur la journée
    - On sélectionne les K meilleurs offsets positifs et on exécute BUY à t0 puis SELL à t0+offset
    """
    t0_total = time.perf_counter()
    equity_curve_times = []
    equity_curve_values = []
    trades = []
    cash = float(initial_cash)

    look_back, num_features, nb_y = _infer_model_shapes(model)

    idx = day_aligned_df.index
    days = idx.normalize().unique()

    # Préparation des fenêtres d'inférence par jour (batch unique)
    t_feat0 = time.perf_counter()
    day_infos = []
    X_list = []
    for day in days:
        mask = (idx.normalize() == day)
        day_df = day_aligned_df.loc[mask]
        if day_df.shape[0] <= look_back + max(2, nb_y):
            continue
        feats = _build_features(day_df, num_features)
        window = feats[:look_back]
        X_list.append(window)
        day_infos.append({'day_df': day_df, 't0_i': (look_back - 1)})
    t_feat1 = time.perf_counter()

    if not X_list:
        t1_total = time.perf_counter()
        return {
            'equity_times': [],
            'equity_values': [],
            'trades': [],
            'final_value': float(cash),
            'perf': {
                'prep_s': 0.0,
                'loop_s': float(t1_total - t0_total),
                'feature_build_s': float(t_feat1 - t_feat0),
                'num_days_total': int(len(days)),
                'num_days_used': 0,
            }
        }

    X_batch = np.stack(X_list, axis=0)

    # Prédiction batch
    t_pred0 = time.perf_counter()
    try:
        y_pred_batch = model.predict(X_batch, verbose=0)
    except Exception:
        y_pred_batch = model(X_batch, training=False).numpy()
    t_pred1 = time.perf_counter()

    # Application des décisions et construction des trades/équity
    t_trd0 = time.perf_counter()
    y_pred_batch = np.asarray(y_pred_batch)
    if y_pred_batch.ndim == 1:
        y_pred_batch = y_pred_batch.reshape(-1, 1)
    for i, info in enumerate(day_infos):
        day_df = info['day_df']
        t0_i = info['t0_i']
        y_pred = y_pred_batch[i].reshape(-1)
        remainder = len(day_df) - look_back
        if remainder <= nb_y:
            continue
        stride_y = remainder // (nb_y + 1)
        offsets = [(j + 1) * stride_y for j in range(nb_y)]
        order = np.argsort(-y_pred)
        k = max(1, int(min(k_trades, nb_y)))
        selected = order[:k]
        t0_ts = day_df.index[t0_i]
        if 'openPrice' in day_df.columns:
            buy_price = float(day_df.iloc[t0_i]['openPrice'])
        else:
            buy_price = float(day_df.iloc[t0_i].select_dtypes(include=[np.number]).iloc[0])
        qty = int(per_trade_amount // max(1e-9, buy_price))
        if qty <= 0:
            continue
        for j in selected:
            off = int(offsets[j])
            sell_i = min(t0_i + off, len(day_df) - 1)
            sell_ts = day_df.index[sell_i]
            if 'openPrice' in day_df.columns:
                sell_price = float(day_df.iloc[sell_i]['openPrice'])
            else:
                sell_price = float(day_df.iloc[sell_i].select_dtypes(include=[np.number]).iloc[0])
            pnl = float((sell_price - buy_price) * qty)
            trades.append({
                'time': str(sell_ts),
                'action': 'SELL',
                'qty': qty,
                'price': sell_price,
                'entry_time': str(t0_ts),
                'entry_price': buy_price,
                'pnl': pnl
            })
            cash += pnl
            equity_curve_times.append(sell_ts)
            equity_curve_values.append(cash)
    t_trd1 = time.perf_counter()

    t1_total = time.perf_counter()
    return {
        'equity_times': [pd.Timestamp(t) for t in equity_curve_times],
        'equity_values': [float(v) for v in equity_curve_values],
        'trades': trades,
        'final_value': float(cash),
        'perf': {
            'prep_s': 0.0,
            'loop_s': float(t1_total - t0_total),
            'feature_build_s': float(t_feat1 - t_feat0),
            'batch_predict_s': float(t_pred1 - t_pred0),
            'trades_build_s': float(t_trd1 - t_trd0),
            'num_days_total': int(len(days)),
            'num_days_used': int(len(day_infos)),
            'look_back': int(look_back),
            'nb_y': int(nb_y)
        }
    }



