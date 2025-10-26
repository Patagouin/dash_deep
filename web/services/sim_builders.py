import pandas as pd
import numpy as np
from dash import html, dash_table
import plotly.graph_objects as go


def build_equity_figure(sim_mode,
                        ref_symbol,
                        trade_symbol,
                        lag,
                        threshold_pct,
                        buy_start_time,
                        sell_end_time,
                        equity_curve_times,
                        equity_curve_values):
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={
            'color': '#FFFFFF'
        },
        title='Simulation — en attente de paramètres',
        uirevision='sim_equity'
    )
    if not equity_curve_times:
        return fig
    fig = go.Figure(
        data=
        [
            go.Scatter(x=equity_curve_times, y=equity_curve_values, mode='lines+markers', name='Equity', line={'color': '#FF8C00'})
        ]
    )
    if sim_mode == 'leadlag':
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font={
                'color': '#FFFFFF'
            },
            title=f"Equity — A={ref_symbol}, B={trade_symbol}, lag={lag} min, seuil={threshold_pct}%",
            uirevision='sim_equity'
        )
    else:
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font={
                'color': '#FFFFFF'
            },
            title=f"Equity — {ref_symbol} | Fenêtre {buy_start_time or '09:30'} → {sell_end_time or '16:00'}",
            uirevision='sim_equity'
        )
    return fig


def build_multi_equity_figure(series_list):
    """Construit une figure avec plusieurs courbes d'equity pour comparaison.
    series_list: liste de dicts { 'label': str, 'times': [...], 'values': [...] }
    """
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={ 'color': '#FFFFFF' },
        title='Comparaison de stratégies — equity',
        uirevision='sim_equity_multi'
    )
    if not series_list:
        return fig
    palette = ['#FF8C00', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, s in enumerate(series_list):
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(x=s.get('times', []), y=s.get('values', []), mode='lines', name=s.get('label', f'strategy_{i+1}'), line={'color': color}))
    return fig


def build_comparison_summary(items):
    """Construit une liste de résumés concaténés pour plusieurs stratégies.
    items: liste de tuples (title, summary_items_list)
    """
    blocks = []
    for title, summary_items in items:
        blocks.append(html.Div([
            html.Div(title, style={'fontWeight': 'bold', 'color': '#FF8C00', 'marginBottom': '4px'}),
            html.Ul([html.Li(it) for it in summary_items], style={ 'color': '#FFFFFF', 'marginTop': '0px'}),
        ], style={'marginBottom': '8px', 'padding': '8px', 'border': '1px solid #333', 'borderRadius': '6px'}) )
    return html.Div(blocks)


def build_daily_outputs(equity_curve_times,
                        equity_curve_values,
                        trades):
    fig_daily = go.Figure()
    fig_daily.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={
            'color': '#FFFFFF'
        },
        title='Equity journalier — en attente de paramètres',
        uirevision='sim_daily'
    )
    daily_table_component = html.Div('')
    avg_daily_return_pct = 0.0
    if not equity_curve_times:
        return fig_daily, daily_table_component, pd.DataFrame(), avg_daily_return_pct
    equity_df = pd.DataFrame(
        {
            'equity': equity_curve_values
        },
        index=pd.to_datetime(equity_curve_times)
    ).sort_index()
    daily_equity = equity_df['equity'].groupby(equity_df.index.date).last()
    if not daily_equity.empty:
        fig_daily = go.Figure(
            data=
            [
                go.Scatter(x=pd.to_datetime(daily_equity.index), y=daily_equity.values, mode='lines+markers', name='Equity (jour)', line={'color': '#FF8C00'})
            ]
        )
        fig_daily.update_layout(
            template='plotly_dark',
            paper_bgcolor='#000000',
            plot_bgcolor='#000000',
            font={
                'color': '#FFFFFF'
            },
            title='Equity (fin de journée)',
            uirevision='sim_daily'
        )

        # Tableau journalier
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame(columns=['time','action','pnl'])
        if not df_trades.empty:
            df_trades['date'] = pd.to_datetime(df_trades['time']).dt.date
            buys_per_day = df_trades[df_trades['action'] == 'BUY'].groupby('date').size()
            sells = df_trades[df_trades['action'] == 'SELL']
            sells_per_day = sells.groupby('date').size()
            pnl_per_day = sells.groupby('date')['pnl'].sum()
        else:
            buys_per_day = pd.Series(dtype=int)
            sells_per_day = pd.Series(dtype=int)
            pnl_per_day = pd.Series(dtype=float)

        df_daily = pd.DataFrame(
            {
                'equity_end': daily_equity
            }
        )
        # Calcul du gain quotidien moyen (%) depuis l'equity
        try:
            daily_ret_series = daily_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            avg_daily_return_pct = float(daily_ret_series.mean() * 100.0) if len(daily_ret_series) > 0 else 0.0
        except Exception:
            avg_daily_return_pct = 0.0

        df_daily.index.name = 'date'
        df_daily['equity_prev'] = df_daily['equity_end'].shift(1)
        df_daily['daily_return_pct'] = (df_daily['equity_end'] / df_daily['equity_prev'] - 1.0) * 100.0
        df_daily['buys'] = buys_per_day
        df_daily['sells'] = sells_per_day
        df_daily['realized_pnl_day'] = pnl_per_day
        df_daily = df_daily.fillna(0)
        df_daily['daily_return_pct'] = df_daily['daily_return_pct'].replace([np.inf, -np.inf], 0)
        df_daily['cum_pnl'] = df_daily['realized_pnl_day'].cumsum()
        df_daily_reset = df_daily.reset_index()

        daily_columns = [
            {
                'name': 'date',
                'id': 'date'
            },
            {
                'name': 'equity_end',
                'id': 'equity_end'
            },
            {
                'name': 'daily_return_%',
                'id': 'daily_return_pct'
            },
            {
                'name': 'buys',
                'id': 'buys'
            },
            {
                'name': 'sells',
                'id': 'sells'
            },
            {
                'name': 'realized_pnl_day',
                'id': 'realized_pnl_day'
            },
            {
                'name': 'cum_pnl',
                'id': 'cum_pnl'
            }
        ]
        daily_table_component = dash_table.DataTable(
            data=df_daily_reset.to_dict('records'),
            columns=daily_columns,
            style_table=
            {
                'overflowX': 'auto',
                'maxHeight': '40vh',
                'overflowY': 'auto'
            },
            style_cell=
            {
                'textAlign': 'center',
                'minWidth': '80px',
                'width': '120px',
                'maxWidth': '200px'
            },
            style_header=
            {
                'backgroundColor': '#000000',
                'color': '#ffffff'
            },
            style_data=
            {
                'backgroundColor': '#1E1E1E',
                'color': '#ffffff'
            }
        )
    else:
        df_daily_reset = pd.DataFrame()

    return fig_daily, daily_table_component, df_daily_reset, avg_daily_return_pct


def build_trades_table(trades):
    if not trades:
        return html.Div('Aucune transaction effectuée sur la période.')
    df_trades = pd.DataFrame(trades)
    columns = [
        { 'name': 'time', 'id': 'time' },
        { 'name': 'action', 'id': 'action' },
        { 'name': 'side', 'id': 'side' },
        { 'name': 'qty', 'id': 'qty' },
        { 'name': 'price', 'id': 'price' },
        { 'name': 'entry_time', 'id': 'entry_time' },
        { 'name': 'entry_price', 'id': 'entry_price' },
        { 'name': 'pnl', 'id': 'pnl' },
        { 'name': 'A_price_t0-lag', 'id': 'ref_price_t0_lag' },
        { 'name': 'A_price_t0', 'id': 'ref_price_t0' }
    ]
    return dash_table.DataTable(
        data=df_trades.to_dict('records'),
        columns=columns,
        style_table={ 'overflowX': 'auto', 'maxHeight': '40vh', 'overflowY': 'auto' },
        style_cell={ 'textAlign': 'center', 'minWidth': '80px', 'width': '120px', 'maxWidth': '200px' },
        style_header={ 'backgroundColor': '#000000', 'color': '#ffffff' },
        style_data={ 'backgroundColor': '#1E1E1E', 'color': '#ffffff' }
    )


def build_summary(sim_mode,
                  lag,
                  threshold_pct,
                  buy_start_time,
                  sell_end_time,
                  aligned,
                  initial_cash,
                  final_value,
                  trades,
                  avg_daily_return_pct,
                  perf_fetch=None,
                  perf_align=None,
                  perf_bt=None,
                  fetch_s=None,
                  align_s=None,
                  bt_s=None,
                  perf_build=None):
    realized_sells = [t for t in trades if t.get('action') == 'SELL']
    realized_pnls = [t.get('pnl', 0.0) for t in realized_sells]
    total_pnl = float(np.nansum(realized_pnls)) if realized_pnls else 0.0
    # Win rate historique (sur nb BUY)
    num_trades = int(sum(1 for t in trades if t.get('action') == 'BUY'))
    wins = int(sum(1 for t in realized_pnls if t > 0))
    win_rate = (wins / num_trades * 100.0) if num_trades > 0 else 0.0
    # Win rate non‑zéro (exclut pnl == 0 des SELL)
    non_zero_sells = [t for t in realized_sells if float(t.get('pnl', 0.0)) != 0.0]
    nz_count = len(non_zero_sells)
    nz_wins = sum(1 for t in non_zero_sells if float(t.get('pnl', 0.0)) > 0.0)
    win_rate_nz = (nz_wins / nz_count * 100.0) if nz_count > 0 else 0.0
    ret_pct = ((final_value - float(initial_cash or 10000.0)) / float(initial_cash or 10000.0) * 100.0)
    overlap_minutes = int(aligned.shape[0])
    num_days = int(aligned.index.normalize().nunique())

    summary_items = [
        f"Trades: {num_trades}",
        f"Win rate: {win_rate:.1f}%",
        f"Win rate (non‑zéro): {win_rate_nz:.1f}%",
        f"PnL réalisé: {total_pnl:.2f} €",
        f"Valeur finale: {final_value:.2f} € ({ret_pct:.2f}%)",
        f"Minutes utilisées: {overlap_minutes}",
        f"Jours couverts: {num_days}",
        (f"Durée de détention (lead-lag): {lag} min" if sim_mode=='leadlag' else "Stratégie: Fenêtre horaire"),
        (f"Fenêtre signal A: {lag} min" if sim_mode=='leadlag' else f"Fenêtre quotidienne: {buy_start_time or '09:30'} → {sell_end_time or '16:00'}"),
        f"Gain moyen/jour: {avg_daily_return_pct:.2f}%"
    ]

    # Perfs détaillées si fournies
    if fetch_s is not None and align_s is not None and bt_s is not None:
        summary_items.extend([
            "--- Perf ---",
            f"Fetch: {fetch_s:.3f}s",
            f"Align: {align_s:.3f}s",
            f"Backtest: {bt_s:.3f}s",
        ])
    if perf_bt:
        summary_items.extend([
            f"Backtest prep: {perf_bt.get('prep_s', 0.0):.3f}s",
            f"Backtest loop: {perf_bt.get('loop_s', 0.0):.3f}s",
        ])
        # Détails backtest si disponibles
        details = []
        if 'points_iterated' in perf_bt:
            details.append(f"Points itérés: {perf_bt.get('points_iterated', 0)}")
        if 'days_total' in perf_bt or 'days_with_window' in perf_bt:
            details.append(
                f"Jours: {perf_bt.get('days_total', 0)} (fenêtrés: {perf_bt.get('days_with_window', 0)})"
            )
        if 'buys' in perf_bt or 'sells' in perf_bt:
            details.append(f"Buys/Sells: {perf_bt.get('buys', 0)}/{perf_bt.get('sells', 0)}")
        # Temps par étape
        step_times_map = [
            ('daily_selection_s', 'Sélection journalière'),
            ('build_mask_s', 'Construction masque'),
            ('mtm_s', 'MTM (courbe equity)'),
            ('buy_logic_s', 'Logique BUY'),
            ('sell_logic_s', 'Logique SELL'),
            ('entry_logic_s', 'Logique entrée'),
            ('exit_logic_s', 'Logique sortie'),
        ]
        for k, label in step_times_map:
            if k in perf_bt:
                details.append(f"{label}: {perf_bt.get(k, 0.0):.3f}s")
        if 'loop_per_point_us' in perf_bt:
            details.append(f"Boucle: {perf_bt.get('loop_per_point_us', 0.0):.1f} µs/point")
        if details:
            summary_items.extend(details)
    # Si perfs détaillées des services
    if perf_fetch:
        summary_items.extend([
            "--- Perf détaillée ---",
            f"Fetch total: {perf_fetch.get('total_s',0):.3f}s",
            f"  DB query: {perf_fetch.get('db_query_s',0):.3f}s",
            f"  Update cotations: {perf_fetch.get('update_cotations_s',0):.3f}s (err={perf_fetch.get('update_errors',0)})",
            f"  Get list: {perf_fetch.get('get_list_s',0):.3f}s",
            f"  Build series: {perf_fetch.get('build_series_s',0):.3f}s",
            f"  Series: {perf_fetch.get('series_built',0)}/{perf_fetch.get('symbols',0)} vides={perf_fetch.get('series_empty',0)} points={perf_fetch.get('total_points',0)}",
        ])
    if perf_align:
        summary_items.extend([
            f"Align total: {perf_align.get('total_s',0):.3f}s",
            f"  Concat/dedup: {perf_align.get('concat_dedup_s',0):.3f}s",
            f"  Resample+ffill+bfill: {perf_align.get('resample_ffill_bfill_s',0):.3f}s",
            f"  Fenêtre+dropna: {perf_align.get('window_dropna_s',0):.3f}s",
        ])

    if perf_build:
        summary_items.extend([
            "--- Build UI ---",
            f"Downsample: {perf_build.get('downsample_s',0):.3f}s",
            f"Equity points (total/affichés/step): {perf_build.get('equity_points_total',0)}/{perf_build.get('equity_points_shown',0)}/{perf_build.get('equity_step',1)}",
            f"Figure equity build: {perf_build.get('equity_fig_build_s',0):.3f}s",
            f"Daily outputs: {perf_build.get('daily_outputs_s',0):.3f}s",
            f"Trades table: {perf_build.get('trades_table_s',0):.3f}s",
            f"to_dict(figures): {perf_build.get('fig_to_dict_s',0):.3f}s",
            f"Callback total: {perf_build.get('callback_total_s',0):.3f}s",
        ])

    return summary_items



