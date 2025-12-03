import pandas as pd
import numpy as np


def backtest_lag_correlation(aligned_prices: pd.DataFrame,
                             ref_symbol: str,
                             trade_symbol: str,
                             lag_minutes: int,
                             threshold_pct: float,
                             initial_cash: float,
                             per_trade_amount: float,
                             minutes_before_close: int = 10,
                             signal_window_minutes: int = 30,
                             buy_start_time: str = None,
                             sell_end_time: str = None,
                             direction: str = 'up',
                             spread_pct: float = 0.0):
    import time
    t0 = time.perf_counter()
    price_A = aligned_prices[ref_symbol]
    price_B = aligned_prices[trade_symbol]
    # Signal sur A basé sur une fenêtre égale au lag demandé
    ret_A_signal = price_A.pct_change(periods=int(max(1, lag_minutes)))

    # Cutoff par jour
    norm_index = aligned_prices.index.normalize()
    day_last_idx = aligned_prices.index.to_series().groupby(norm_index).max()
    cutoff_by_day = day_last_idx - pd.Timedelta(minutes=minutes_before_close)
    cutoff_for_rows = norm_index.map(cutoff_by_day)
    is_after_cutoff = pd.Series(aligned_prices.index >= cutoff_for_rows.values, index=aligned_prices.index)

    # Boucle de backtest
    cash = float(initial_cash)
    qty = 0  # >0 long, <0 short
    entry_price = np.nan
    position_open_time = None

    equity_curve_times = []
    equity_curve_values = []
    trades = []

    threshold = threshold_pct / 100.0

    timestamps = aligned_prices.index
    prices_B = price_B.values
    prices_A = price_A.values
    sig_A = ret_A_signal.values
    hold_until_idx = None
    entry_side = None  # 'long' or 'short'

    # Parse fenêtres horaires si fournie (HH:MM)
    def _parse_hhmm(s):
        try:
            h, m = str(s).split(':')
            return int(h), int(m)
        except Exception:
            return None
    buy_hm = _parse_hhmm(buy_start_time) if buy_start_time else None
    sell_hm = _parse_hhmm(sell_end_time) if sell_end_time else None

    points_iterated = 0
    buys = 0
    sells = 0
    dt_mtm = 0.0
    dt_entry = 0.0
    dt_exit = 0.0

    t_loop_start = time.perf_counter()
    for i in range(len(timestamps)):
        ts = timestamps[i]
        pb = prices_B[i]
        rA = sig_A[i]
        after_cut = is_after_cutoff.values[i]

        # Fenêtre quotidienne: autoriser l'entrée seulement si dans [buy_start, sell_end)
        allow_entry_now = True
        if buy_hm and sell_hm:
            cur_h = ts.hour
            cur_m = ts.minute
            allow_entry_now = ((cur_h, cur_m) >= buy_hm) and ((cur_h, cur_m) < sell_hm)

        # MTM
        _t0 = time.perf_counter()
        portfolio_value = cash + (qty * pb)
        equity_curve_times.append(ts)
        equity_curve_values.append(portfolio_value)
        dt_mtm += (time.perf_counter() - _t0)
        points_iterated += 1

        # Signaux directionnels
        has_sig = (rA is not None) and (not np.isnan(rA))
        signal_up = has_sig and (rA >= threshold)
        signal_down = has_sig and (rA <= -threshold)

        # Sortie: fin de période de détention ou fin de séance
        if qty != 0:
            must_exit = False
            if after_cut:
                must_exit = True
            elif hold_until_idx is not None and i >= hold_until_idx:
                must_exit = True
            # Forcer sortie si on dépasse l'heure de fin
            if not must_exit and sell_hm:
                if (ts.hour, ts.minute) >= sell_hm:
                    must_exit = True
            if must_exit:
                _te0 = time.perf_counter()
                mid_exit_price = pb
                # Calcul du spread pour la sortie
                half_spread_mult = spread_pct / 100.0 / 2.0
                
                if qty > 0:
                    # Close long: vendre au bid (mid - half_spread)
                    exit_price = mid_exit_price * (1.0 - half_spread_mult)
                    exit_qty = qty
                    cash += exit_qty * exit_price
                    realized = (exit_price - entry_price) * exit_qty
                    trades.append({
                        'time': ts,
                        'action': 'SELL',
                        'side': 'long',
                        'qty': int(exit_qty),
                        'price': float(exit_price),
                        'mid_price': float(mid_exit_price),
                        'spread_pct': spread_pct,
                        'pnl': float(realized),
                        'entry_time': position_open_time,
                        'entry_price': float(entry_price)
                    })
                else:
                    # Close short: racheter à l'ask (mid + half_spread)
                    exit_price = mid_exit_price * (1.0 + half_spread_mult)
                    exit_qty = -qty
                    cash -= exit_qty * exit_price
                    realized = (entry_price - exit_price) * exit_qty
                    trades.append({
                        'time': ts,
                        'action': 'SELL',
                        'side': 'short',
                        'qty': int(exit_qty),
                        'price': float(exit_price),
                        'mid_price': float(mid_exit_price),
                        'spread_pct': spread_pct,
                        'pnl': float(realized),
                        'entry_time': position_open_time,
                        'entry_price': float(entry_price)
                    })
                qty = 0
                entry_price = np.nan
                position_open_time = None
                hold_until_idx = None
                sells += 1
                dt_exit += (time.perf_counter() - _te0)
                if after_cut:
                    continue

        # Entrée: selon direction sélectionnée et pas après cutoff
        if qty == 0 and (not after_cut) and allow_entry_now and per_trade_amount > 0:
            want_long = False
            want_short = False
            if direction == 'up':
                want_long = signal_up
            elif direction == 'down':
                want_short = signal_down
            elif direction == 'both':
                want_long = signal_up
                want_short = (not want_long) and signal_down
            # Calcul du spread: half_spread pour ask/bid
            half_spread_mult = spread_pct / 100.0 / 2.0
            
            if want_long:
                _tb0 = time.perf_counter()
                mid_price = pb
                # Prix d'achat = prix mid + half_spread (ask)
                price = mid_price * (1.0 + half_spread_mult)
                buy_qty = int(np.floor(per_trade_amount / max(1e-12, price)))
                if buy_qty > 0 and (cash >= buy_qty * price):
                    cash -= buy_qty * price
                    qty = buy_qty
                    entry_price = price
                    position_open_time = ts
                    hold_until_idx = i + int(max(1, lag_minutes))
                    # Prix de l'action A à t0-lag et t0 (pour vérification du seuil)
                    _lagp = int(max(1, lag_minutes))
                    a_t0 = float(prices_A[i]) if i < len(prices_A) else None
                    a_t0_lag = float(prices_A[i - _lagp]) if (i - _lagp) >= 0 else None
                    trades.append({
                        'time': ts,
                        'action': 'BUY',
                        'side': 'long',
                        'qty': int(buy_qty),
                        'price': float(price),
                        'mid_price': float(mid_price),
                        'spread_pct': spread_pct,
                        'ref_price_t0_lag': a_t0_lag,
                        'ref_price_t0': a_t0
                    })
                    buys += 1
                dt_entry += (time.perf_counter() - _tb0)
            elif want_short:
                _tb0 = time.perf_counter()
                mid_price = pb
                # Prix de vente à découvert = prix mid - half_spread (bid)
                price = mid_price * (1.0 - half_spread_mult)
                sell_qty = int(np.floor(per_trade_amount / max(1e-12, mid_price)))
                if sell_qty > 0:
                    cash += sell_qty * price
                    qty = -sell_qty
                    entry_price = price
                    position_open_time = ts
                    hold_until_idx = i + int(max(1, lag_minutes))
                    _lagp = int(max(1, lag_minutes))
                    a_t0 = float(prices_A[i]) if i < len(prices_A) else None
                    a_t0_lag = float(prices_A[i - _lagp]) if (i - _lagp) >= 0 else None
                    trades.append({
                        'time': ts,
                        'action': 'BUY',
                        'side': 'short',
                        'qty': int(sell_qty),
                        'price': float(price),
                        'mid_price': float(mid_price),
                        'spread_pct': spread_pct,
                        'ref_price_t0_lag': a_t0_lag,
                        'ref_price_t0': a_t0
                    })
                    buys += 1
                dt_entry += (time.perf_counter() - _tb0)

    # Close at end if open
    if qty != 0:
        mid_last_price = price_B.iloc[-1]
        half_spread_mult = spread_pct / 100.0 / 2.0
        
        if qty > 0:
            # Close long: vendre au bid
            last_price = mid_last_price * (1.0 - half_spread_mult)
            cash += qty * last_price
            realized = (last_price - entry_price) * qty
            trades.append({
                'time': aligned_prices.index[-1],
                'action': 'SELL',
                'side': 'long',
                'qty': int(qty),
                'price': float(last_price),
                'mid_price': float(mid_last_price),
                'spread_pct': spread_pct,
                'pnl': float(realized),
                'entry_time': position_open_time,
                'entry_price': float(entry_price)
            })
        else:
            # Close short: racheter à l'ask
            last_price = mid_last_price * (1.0 + half_spread_mult)
            exit_qty = -qty
            cash -= exit_qty * last_price
            realized = (entry_price - last_price) * exit_qty
            trades.append({
                'time': aligned_prices.index[-1],
                'action': 'SELL',
                'side': 'short',
                'qty': int(exit_qty),
                'price': float(last_price),
                'mid_price': float(mid_last_price),
                'spread_pct': spread_pct,
                'pnl': float(realized),
                'entry_time': position_open_time,
                'entry_price': float(entry_price)
            })
        qty = 0
        entry_price = np.nan
        position_open_time = None

    t_loop_end = time.perf_counter()
    final_portfolio_value = cash
    if equity_curve_times:
        equity_curve_times.append(aligned_prices.index[-1])
        equity_curve_values.append(final_portfolio_value)

    return {
        'equity_times': equity_curve_times,
        'equity_values': equity_curve_values,
        'trades': trades,
        'final_value': final_portfolio_value,
        'perf': {
            'prep_s': float(t_loop_start - t0),
            'loop_s': float(t_loop_end - t_loop_start),
            'points_iterated': int(points_iterated),
            'buys': int(buys),
            'sells': int(sells),
            'spread_pct': float(spread_pct),
            'mtm_s': float(dt_mtm),
            'entry_logic_s': float(dt_entry),
            'exit_logic_s': float(dt_exit),
            'loop_per_point_us': float(((t_loop_end - t_loop_start) / max(1, points_iterated)) * 1e6),
        }
    }


def backtest_time_window(aligned_prices: pd.DataFrame,
                         symbol: str,
                         initial_cash: float,
                         per_trade_amount: float,
                         buy_start_time: str,
                         sell_end_time: str,
                         spread_pct: float = 0.0):
    """
    Stratégie simple: acheter chaque jour à buy_start_time, vendre à sell_end_time.
    Si buy/sell exacts non présents, utiliser le dernier prix disponible avant/à l'heure.
    
    Args:
        spread_pct: Spread bid-ask en pourcentage (ex: 0.1 = 0.1%)
    """
    import time
    t0 = time.perf_counter()
    price = aligned_prices[symbol]
    idx = aligned_prices.index
    norm_index = idx.normalize()
    days = norm_index.unique()

    def _parse_hhmm(s):
        h, m = str(s).split(':')
        return int(h), int(m)

    buy_h, buy_m = _parse_hhmm(buy_start_time)
    sell_h, sell_m = _parse_hhmm(sell_end_time)

    cash = float(initial_cash)
    qty = 0
    equity_curve_times = []
    equity_curve_values = []
    trades = []

    days_total = 0
    days_with_window = 0
    points_iterated = 0
    buys = 0
    sells = 0
    dt_daily_sel = 0.0
    dt_build_mask = 0.0
    dt_mtm = 0.0
    dt_buy = 0.0
    dt_sell = 0.0

    t_loop_start = time.perf_counter()
    for day in days:
        day_mask = (norm_index == day)
        day_times = idx[day_mask]
        if len(day_times) == 0:
            continue
        days_total += 1
        # Trouver timestamps d'achat/vente pour ce jour
        _tsel0 = time.perf_counter()
        buy_candidates = [t for t in day_times if (t.hour, t.minute) >= (buy_h, buy_m)]
        sell_candidates = [t for t in day_times if (t.hour, t.minute) >= (sell_h, sell_m)]
        dt_daily_sel += (time.perf_counter() - _tsel0)
        if not buy_candidates or not sell_candidates:
            # Mettre à jour equity sans trade (vectorisé)
            _tmtm0 = time.perf_counter()
            prices_day = price.loc[day_times].to_numpy()
            vals = (cash + qty * prices_day)
            equity_curve_times.extend(day_times)
            equity_curve_values.extend(vals.tolist())
            dt_mtm += (time.perf_counter() - _tmtm0)
            points_iterated += int(len(day_times))
            continue
        buy_ts = buy_candidates[0]
        sell_ts = sell_candidates[0]
        if sell_ts <= buy_ts:
            # fenêtre invalide ce jour (vectorisé)
            _tmtm0 = time.perf_counter()
            prices_day = price.loc[day_times].to_numpy()
            vals = (cash + qty * prices_day)
            equity_curve_times.extend(day_times)
            equity_curve_values.extend(vals.tolist())
            dt_mtm += (time.perf_counter() - _tmtm0)
            points_iterated += int(len(day_times))
            continue

        # Parcourir uniquement la fenêtre utile [buy_ts -> sell_ts]
        times_iter = day_times
        if 'buy_ts' in locals() and 'sell_ts' in locals() and sell_ts > buy_ts:
            _tmask0 = time.perf_counter()
            mask = (day_times >= buy_ts) & (day_times <= sell_ts)
            if mask.any():
                times_iter = day_times[mask]
            dt_build_mask += (time.perf_counter() - _tmask0)
            days_with_window += 1

        # Vectorisation de la fenêtre utile [buy_ts -> sell_ts]
        if times_iter.size > 0:
            # Indices relatifs dans la journée
            pos_buy = day_times.get_loc(buy_ts)
            pos_sell = day_times.get_loc(sell_ts)
            window_times = day_times[pos_buy:pos_sell + 1]
            prices_window = price.loc[window_times].to_numpy()

            # BUY au début de fenêtre si possible
            _tb0 = time.perf_counter()
            mid_p_buy = float(prices_window[0])
            half_spread_mult = spread_pct / 100.0 / 2.0
            # Prix d'achat = mid + half_spread (ask)
            p_buy = mid_p_buy * (1.0 + half_spread_mult)
            buy_qty = int(np.floor(per_trade_amount / max(1e-12, p_buy)))
            if buy_qty > 0 and (cash >= buy_qty * p_buy):
                cash -= buy_qty * p_buy
                qty = buy_qty
                trades.append({
                    'time': window_times[0], 
                    'action': 'BUY', 
                    'qty': int(buy_qty), 
                    'price': float(p_buy),
                    'mid_price': float(mid_p_buy),
                    'spread_pct': spread_pct
                })
                buys += 1
            dt_buy += (time.perf_counter() - _tb0)

            # MTM vectorisé sur la fenêtre
            _tmtm0 = time.perf_counter()
            vals = (cash + qty * prices_window)
            equity_curve_times.extend(window_times)
            equity_curve_values.extend(vals.tolist())
            dt_mtm += (time.perf_counter() - _tmtm0)
            points_iterated += int(len(window_times))

            # SELL à la fin de fenêtre si position ouverte
            if qty > 0:
                _ts0 = time.perf_counter()
                mid_p_sell = float(prices_window[-1])
                # Prix de vente = mid - half_spread (bid)
                p_sell = mid_p_sell * (1.0 - half_spread_mult)
                cash += qty * p_sell
                realized = (p_sell - p_buy) * qty if trades and trades[-1]['action'] == 'BUY' else 0.0
                trades.append({
                    'time': window_times[-1], 
                    'action': 'SELL', 
                    'qty': int(qty), 
                    'price': float(p_sell),
                    'mid_price': float(mid_p_sell),
                    'spread_pct': spread_pct,
                    'pnl': float(realized)
                })
                qty = 0
                sells += 1
                dt_sell += (time.perf_counter() - _ts0)

    # Close any open at last available price
    if qty > 0:
        mid_last_price = price.iloc[-1]
        half_spread_mult = spread_pct / 100.0 / 2.0
        last_price = mid_last_price * (1.0 - half_spread_mult)
        cash += qty * last_price
        realized = (last_price - trades[-1]['price']) * qty if trades and trades[-1]['action'] == 'BUY' else 0.0
        trades.append({
            'time': idx[-1], 
            'action': 'SELL', 
            'qty': int(qty), 
            'price': float(last_price),
            'mid_price': float(mid_last_price),
            'spread_pct': spread_pct,
            'pnl': float(realized)
        })
        qty = 0

    t_loop_end = time.perf_counter()
    final_portfolio_value = cash
    if equity_curve_times:
        equity_curve_times.append(idx[-1])
        equity_curve_values.append(final_portfolio_value)

    return {
        'equity_times': equity_curve_times,
        'equity_values': equity_curve_values,
        'trades': trades,
        'final_value': final_portfolio_value,
        'perf': {
            'prep_s': float(t_loop_start - t0),
            'loop_s': float(t_loop_end - t_loop_start),
            'days_total': int(days_total),
            'days_with_window': int(days_with_window),
            'points_iterated': int(points_iterated),
            'buys': int(buys),
            'sells': int(sells),
            'daily_selection_s': float(dt_daily_sel),
            'build_mask_s': float(dt_build_mask),
            'mtm_s': float(dt_mtm),
            'buy_logic_s': float(dt_buy),
            'sell_logic_s': float(dt_sell),
            'loop_per_point_us': float(((t_loop_end - t_loop_start) / max(1, points_iterated)) * 1e6),
        }
    }


