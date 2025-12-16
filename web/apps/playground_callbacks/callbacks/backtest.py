# -*- coding: utf-8 -*-
"""
Callbacks de backtest pour le playground.
Migré depuis `web/apps/playground.py`.
"""

import logging
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, html

from app import app
from web.apps.model_config import (
    DEFAULT_INITIAL_CASH,
    DEFAULT_K_TRADES,
    DEFAULT_LOOK_BACK,
    DEFAULT_NB_Y,
    DEFAULT_SPREAD_PCT,
    DEFAULT_TRADE_AMOUNT,
)
from web.services.sim_builders import build_equity_figure


@app.callback(
    [
        Output('play_equity_graph', 'figure'),
        Output('play_trades_table', 'children'),
        Output('play_summary', 'children'),
    ],
    [Input('play_run_backtest', 'n_clicks')],
    [
        State('play_df_store', 'data'),
        State('play_predictions_store', 'data'),
        State('play_initial_cash', 'value'),
        State('play_trade_amount', 'value'),
        State('play_k_trades', 'value'),
        State('play_spread_pct', 'value'),
        State('play_strategy', 'value'),
    ],
    prevent_initial_call=True,
)
def run_backtest(n_clicks, store_json, predictions_data, initial_cash, per_trade, k_trades, spread_pct, strategy):
    """
    Exécute le backtest basé sur les prédictions stockées.

    Stratégies:
    - 'long': Acheter si hausse prédite (BUY → SELL)
    - 'short': Vendre si baisse prédite (SELL → BUY, short selling)
    - 'both': Long si hausse, Short si baisse
    """
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        font={'color': '#FFF'},
        title='Équité — en attente',
        height=400,
        uirevision='play_equity',
    )

    if not n_clicks or not store_json or not predictions_data:
        return empty_fig, html.Div("Entraînez d'abord un modèle"), html.Div('')

    try:
        df = pd.read_json(StringIO(store_json), orient='split')
        initial_cash_val = float(initial_cash or DEFAULT_INITIAL_CASH)
        per_trade_val = float(per_trade or DEFAULT_TRADE_AMOUNT)
        k_trades_val = int(k_trades or DEFAULT_K_TRADES)
        spread_pct_val = float(spread_pct or DEFAULT_SPREAD_PCT)
        strategy_val = strategy or 'long'

        y_pred_test = predictions_data.get('y_pred_test')
        look_back = predictions_data.get('look_back', DEFAULT_LOOK_BACK)
        nb_y = predictions_data.get('nb_y', DEFAULT_NB_Y)
        pred_type = predictions_data.get('prediction_type', 'return')

        if not y_pred_test:
            return empty_fig, html.Div('Pas de prédictions disponibles'), html.Div('')

        logging.info(
            "[Backtest] Démarrage: %s prédictions, K=%s/jour, stratégie=%s",
            len(y_pred_test),
            k_trades_val,
            strategy_val,
        )

        baseline = 1.0 if pred_type == 'price' else 0.0

        equity_curve_times = []
        equity_curve_values = []
        trades = []
        cash = initial_cash_val

        idx = df.index
        days = idx.normalize().unique()
        split_idx = int(len(days) * 0.8)
        test_days = days[split_idx:]

        pred_idx = 0
        for day in test_days:
            if pred_idx >= len(y_pred_test):
                break

            mask = idx.normalize() == day
            day_df = df.loc[mask]
            if len(day_df) <= look_back + nb_y:
                continue

            y_pred_day = y_pred_test[pred_idx]
            pred_idx += 1
            y_pred_array = np.array(y_pred_day)

            remainder = len(day_df) - look_back
            stride_y = max(1, remainder // (nb_y + 1))
            offsets = [(j + 1) * stride_y for j in range(nb_y)]

            candidates = []
            for j in range(len(y_pred_array)):
                pred_value = float(y_pred_array[j])
                is_up = pred_value > baseline
                is_down = pred_value < baseline
                amplitude = abs(pred_value - baseline)

                if strategy_val == 'long' and is_up:
                    candidates.append((j, 'LONG', amplitude, pred_value))
                elif strategy_val == 'short' and is_down:
                    candidates.append((j, 'SHORT', amplitude, pred_value))
                elif strategy_val == 'both':
                    if is_up:
                        candidates.append((j, 'LONG', amplitude, pred_value))
                    elif is_down:
                        candidates.append((j, 'SHORT', amplitude, pred_value))

            candidates.sort(key=lambda x: -x[2])

            day_trades = []
            occupied_ranges = []

            for j, direction, amplitude, pred_value in candidates:
                _ = amplitude
                if len(day_trades) >= k_trades_val:
                    break

                entry_idx = look_back
                off = int(offsets[j]) if j < len(offsets) else stride_y * (j + 1)
                exit_idx = min(entry_idx + off, len(day_df) - 1)

                overlaps = False
                for (occ_entry, occ_exit) in occupied_ranges:
                    if not (exit_idx <= occ_entry or entry_idx >= occ_exit):
                        overlaps = True
                        break

                if overlaps:
                    continue

                occupied_ranges.append((entry_idx, exit_idx))

                entry_time = day_df.index[entry_idx]
                exit_time = day_df.index[exit_idx]
                mid_entry_price = float(day_df.iloc[entry_idx]['openPrice'])
                mid_exit_price = float(day_df.iloc[exit_idx]['openPrice'])

                half_spread = spread_pct_val / 100.0 / 2.0

                qty = int(per_trade_val // max(1e-9, mid_entry_price))
                if qty <= 0:
                    continue

                if direction == 'LONG':
                    entry_price = mid_entry_price * (1 + half_spread)
                    exit_price = mid_exit_price * (1 - half_spread)
                    pnl = float((exit_price - entry_price) * qty)
                else:
                    entry_price = mid_entry_price * (1 - half_spread)
                    exit_price = mid_exit_price * (1 + half_spread)
                    pnl = float((entry_price - exit_price) * qty)

                day_trades.append(
                    {
                        'entry_time': str(entry_time),
                        'exit_time': str(exit_time),
                        'direction': direction,
                        'qty': qty,
                        'entry_price': round(entry_price, 4),
                        'exit_price': round(exit_price, 4),
                        'predicted': round(pred_value, 6),
                        'pnl': round(pnl, 2),
                    }
                )

                cash += pnl
                equity_curve_times.append(exit_time)
                equity_curve_values.append(cash)

            trades.extend(day_trades)

        logging.info("[Backtest] Terminé: %s trades, Cash final: %.2f€", len(trades), cash)

        if equity_curve_times:
            eq_fig = build_equity_figure(
                'model',
                'SYNTH',
                None,
                None,
                None,
                None,
                None,
                equity_curve_times,
                equity_curve_values,
            )
        else:
            eq_fig = go.Figure()
            eq_fig.add_trace(
                go.Scatter(
                    x=[df.index[0], df.index[-1]],
                    y=[initial_cash_val, initial_cash_val],
                    mode='lines',
                    name='Cash initial',
                )
            )

        pct_return = ((cash / initial_cash_val) - 1) * 100
        eq_fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000',
            plot_bgcolor='#000',
            font={'color': '#FFF'},
            title=f'💰 Équité: {cash:,.2f}€ ({pct_return:+.2f}%) — {len(trades)} trades',
            height=400,
            uirevision='play_equity',
        )

        trades_table = _build_trades_table_v2(trades)

        total_pnl = cash - initial_cash_val
        num_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        strategy_labels = {'long': 'LONG (hausse)', 'short': 'SHORT (baisse)', 'both': 'LONG & SHORT'}
        long_count = len([t for t in trades if t.get('direction') == 'LONG'])
        short_count = len([t for t in trades if t.get('direction') == 'SHORT'])

        summary_items = [
            f"💰 Capital final: {cash:,.2f}€",
            f"📊 P&L total: {total_pnl:+,.2f}€ ({pct_return:+.2f}%)",
            f"📈 Trades: {num_trades} — Win rate: {win_rate:.1f}%",
            f"🎯 Stratégie: {strategy_labels.get(strategy_val, strategy_val)} (📈{long_count} / 📉{short_count})",
            f"💵 Spread: {spread_pct_val:.2f}% — K={k_trades_val}/jour",
        ]
        summary = html.Ul([html.Li(it) for it in summary_items], style={'color': '#FFFFFF'})

        return eq_fig, trades_table, summary

    except Exception as e:
        logging.error("[Backtest] Erreur: %s", e)
        empty_fig.update_layout(title=f'❌ Erreur backtest: {e}')
        return empty_fig, html.Div(f'Erreur: {e}'), html.Div('')


def _build_trades_table_v2(trades):
    """Construit un tableau de trades avec direction, heures d'entrée/sortie."""
    if not trades:
        return html.Div('Aucun trade effectué', style={'color': '#888', 'padding': '8px'})

    rows = []
    for i, t in enumerate(trades[-30:], 1):
        pnl = t.get('pnl', 0)
        pnl_color = '#4CAF50' if pnl > 0 else '#f44336' if pnl < 0 else '#888'
        direction = t.get('direction', 'LONG')

        dir_color = '#4CAF50' if direction == 'LONG' else '#f44336'
        dir_icon = '📈' if direction == 'LONG' else '📉'

        entry_time = t.get('entry_time', '-')
        exit_time = t.get('exit_time', '-')

        entry_dt = entry_time[:10] if len(entry_time) >= 10 else '-'
        entry_hr = entry_time[11:16] if len(entry_time) >= 16 else '-'
        exit_hr = exit_time[11:16] if len(exit_time) >= 16 else '-'

        rows.append(
            html.Tr(
                [
                    html.Td(str(i), style={'padding': '4px 6px', 'textAlign': 'center'}),
                    html.Td(
                        f"{dir_icon}",
                        style={
                            'padding': '4px 6px',
                            'textAlign': 'center',
                            'color': dir_color,
                            'fontSize': '14px',
                        },
                    ),
                    html.Td(entry_dt, style={'padding': '4px 6px'}),
                    html.Td(entry_hr, style={'padding': '4px 6px', 'textAlign': 'center'}),
                    html.Td(exit_hr, style={'padding': '4px 6px', 'textAlign': 'center'}),
                    html.Td(f"{t.get('qty', 0)}", style={'padding': '4px 6px', 'textAlign': 'right'}),
                    html.Td(
                        f"{t.get('entry_price', 0):.2f}",
                        style={'padding': '4px 6px', 'textAlign': 'right'},
                    ),
                    html.Td(
                        f"{t.get('exit_price', 0):.2f}",
                        style={'padding': '4px 6px', 'textAlign': 'right'},
                    ),
                    html.Td(
                        f"{pnl:+.2f}€",
                        style={
                            'padding': '4px 6px',
                            'textAlign': 'right',
                            'color': pnl_color,
                            'fontWeight': 'bold',
                        },
                    ),
                ]
            )
        )

    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th('#', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center'}),
                        html.Th('Dir', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center'}),
                        html.Th('Date', style={'padding': '4px 6px', 'borderBottom': '1px solid #444'}),
                        html.Th('Entrée', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center'}),
                        html.Th('Sortie', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'center'}),
                        html.Th('Qté', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right'}),
                        html.Th('P.Entrée', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right'}),
                        html.Th('P.Sortie', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right'}),
                        html.Th('P&L', style={'padding': '4px 6px', 'borderBottom': '1px solid #444', 'textAlign': 'right'}),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        style={'width': '100%', 'color': '#FFF', 'fontSize': '11px', 'borderCollapse': 'collapse'},
    )

