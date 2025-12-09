"""
Fonctions de construction des graphiques pour le Playground.
"""

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from io import StringIO
from typing import Optional, List, Dict, Tuple


def build_segments_graph(
    store_json: str,
    look_back: int,
    stride: int,
    first_minutes: int,
    predictions: Optional[List] = None,
    nb_y: Optional[int] = None,
    predictions_train: Optional[List] = None,
    prediction_type: str = 'return',
    extra_predictions: Optional[List[Dict]] = None
) -> go.Figure:
    """
    Construit le graphique des segments train/test avec prédictions.
    
    Args:
        store_json: JSON contenant les données
        look_back: Taille de la fenêtre
        stride: Pas d'échantillonnage
        first_minutes: Minutes d'observation
        predictions: Prédictions sur le test set
        nb_y: Nombre de points Y
        predictions_train: Prédictions sur le train set
        prediction_type: Type de prédiction
        extra_predictions: Prédictions supplémentaires
    
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000',
        plot_bgcolor='#000',
        font={'color': '#FFF'},
        title='Segments entraînement / test',
        height=320,
        uirevision='play_segments'
    )
    
    if not store_json:
        return fig
    
    try:
        df = pd.read_json(StringIO(store_json), orient='split')
    except Exception:
        return fig
    
    if df is None or df.empty:
        return fig
    
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(how='any')
    except Exception:
        pass
    
    if df.index.dtype == object:
        return fig
    
    idx = df.index
    norm = idx.normalize()
    days = norm.unique()
    
    if len(days) == 0:
        return fig
    
    # Série complète en fond
    try:
        fig.add_trace(go.Scatter(
            x=idx, y=df['openPrice'].values,
            mode='lines', name='Série',
            line={'color': '#888888', 'width': 1},
            opacity=0.35
        ))
    except Exception:
        pass
    
    split_idx = int(len(days) * 0.8)
    split_day = days[split_idx - 1] if split_idx > 0 else days[0]
    
    n = len(df)
    masks = {
        'train_obs': np.zeros(n, dtype=bool),
        'train_rest': np.zeros(n, dtype=bool),
        'test_obs': np.zeros(n, dtype=bool),
        'test_rest': np.zeros(n, dtype=bool),
    }
    
    obs_len_steps = int(max(1, first_minutes or 60))
    
    for d in days:
        day_mask = (norm == d)
        pos = np.where(day_mask)[0]
        if pos.size == 0:
            continue
        
        obs_len = min(obs_len_steps, pos.size)
        obs_idx = pos[:obs_len]
        rest_idx = pos[obs_len:]
        
        if d <= split_day:
            masks['train_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['train_rest'][rest_idx] = True
        else:
            masks['test_obs'][obs_idx] = True
            if rest_idx.size > 0:
                masks['test_rest'][rest_idx] = True
    
    def add_series(name, mask, color, width=2):
        y = np.where(mask, df['openPrice'].values, np.nan)
        fig.add_trace(go.Scatter(
            x=idx, y=y, mode='lines', name=name,
            line={'color': color, 'width': width}
        ))
    
    add_series('Train (premières min)', masks['train_obs'], '#1f77b4', 2)
    add_series('Train (reste)', masks['train_rest'], '#2ca02c', 2)
    add_series('Test (premières min)', masks['test_obs'], '#9467bd', 2)
    add_series('Test (reste)', masks['test_rest'], '#d62728', 2)

    # Reconstruction des prédictions
    def reconstruct_predictions(predictions_data, day_list, color_name, color_hex):
        if predictions_data is not None and len(predictions_data) > 0:
            try:
                pred_idx_flat = []
                pred_values_flat = []
                preds_array = np.array(predictions_data) if isinstance(predictions_data, list) else predictions_data
                preds_flat = preds_array.flatten()
                
                pred_idx_in_flat = 0
                for day in day_list:
                    day_mask = (norm == day)
                    pos = np.where(day_mask)[0]
                    if pos.size == 0:
                        continue
                    
                    obs_len = min(obs_len_steps, pos.size)
                    if obs_len >= pos.size:
                        continue
                    
                    rest_pos = pos[obs_len:]
                    if len(rest_pos) == 0:
                        continue
                    
                    remainder = len(rest_pos)
                    nb_y_used = nb_y if nb_y is not None and nb_y > 0 else min(5, remainder)
                    
                    remaining_preds = len(preds_flat) - pred_idx_in_flat
                    if remaining_preds < nb_y_used:
                        nb_y_used = remaining_preds
                    if nb_y_used <= 0:
                        continue
                    
                    base_price = float(df.iloc[pos[obs_len - 1]]['openPrice'])
                    if base_price == 0:
                        continue
                    
                    stride_y = remainder // (nb_y_used + 1) if nb_y_used > 0 else 1
                    offsets = [(j + 1) * stride_y for j in range(min(nb_y_used, remainder))]
                    
                    current_pred_price = base_price
                    for i in range(nb_y_used):
                        if pred_idx_in_flat >= len(preds_flat):
                            break
                        if i < len(offsets) and offsets[i] < len(rest_pos):
                            off = offsets[i]
                            pred_val = float(preds_flat[pred_idx_in_flat])
                            
                            if prediction_type == 'price':
                                current_pred_price = base_price * pred_val
                            elif prediction_type == 'signal':
                                cls = int(pred_val)
                                current_pred_price = base_price * (1.0 + (cls - 2) * 0.01)
                            else:
                                current_pred_price = current_pred_price * (1.0 + pred_val)
                                
                            pred_idx_flat.append(idx[rest_pos[off]])
                            pred_values_flat.append(current_pred_price)
                            pred_idx_in_flat += 1
                
                if pred_values_flat:
                    fig.add_trace(go.Scatter(
                        x=pred_idx_flat, y=pred_values_flat,
                        mode='lines+markers', name=color_name,
                        line={'color': color_hex, 'width': 2},
                        marker={'size': 4}
                    ))
            except Exception:
                pass

    train_days = days[:split_idx]
    test_days = days[split_idx:]
    
    reconstruct_predictions(predictions_train, train_days, 'Prédiction (train)', '#17becf')
    reconstruct_predictions(predictions, test_days, 'Prédiction (test)', '#FF8C00')
    
    if extra_predictions:
        for extra in extra_predictions:
            if 'train' in extra:
                reconstruct_predictions(extra['train'], train_days, f"{extra['name']} (train)", extra.get('color', '#888888'))
            if 'test' in extra:
                reconstruct_predictions(extra['test'], test_days, f"{extra['name']} (test)", extra.get('color', '#FF8C00'))
    
    return fig


def build_generalization_figure(
    df: pd.DataFrame,
    sample_days: List,
    obs_window: int,
    y_pred: np.ndarray,
    nb_y: int,
    prediction_type: str = 'return'
) -> Tuple[go.Figure, float, float, int]:
    """
    Construit un graphe pour visualiser la généralisation du modèle.
    
    Returns:
        Tuple (figure, MAE, RMSE, nb_points)
    """
    fig = go.Figure()
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={'color': '#FFFFFF'},
        title='Test de généralisation — courbe actuelle',
        height=420,
        uirevision='play_generalization'
    )

    if df is None or df.empty or y_pred is None or len(sample_days) == 0:
        return fig, 0.0, 0.0, 0

    idx = df.index
    norm = idx.normalize()

    try:
        fig.add_trace(go.Scatter(
            x=idx, y=df['openPrice'].values,
            mode='lines', name='Série (nouvelle)',
            line={'color': '#FF8C00', 'width': 1.8},
            opacity=0.55
        ))
    except Exception:
        pass

    all_x = []
    all_true = []
    all_pred = []

    for i, d in enumerate(sample_days):
        if i >= y_pred.shape[0]:
            break

        day_mask = (norm == d)
        pos = np.where(day_mask)[0]
        if pos.size == 0:
            continue

        obs_len = min(int(obs_window), pos.size)
        if obs_len >= pos.size:
            continue

        rest_pos = pos[obs_len:]
        remainder = len(rest_pos)
        if remainder <= nb_y:
            continue

        stride_y = remainder // (nb_y + 1)
        if stride_y <= 0:
            continue

        offsets = [(j + 1) * stride_y for j in range(nb_y)]

        base_index = pos[obs_len - 1]
        base_price = float(df.iloc[base_index]['openPrice'])
        if base_price == 0:
            continue

        current_pred_price = base_price
        y_pred_vec = np.array(y_pred[i]).flatten()

        for j, off in enumerate(offsets):
            if j >= len(y_pred_vec):
                break
            if off >= len(rest_pos):
                break

            true_index = rest_pos[off]
            true_price = float(df.iloc[true_index]['openPrice'])
            pred_val = float(y_pred_vec[j])

            if prediction_type == 'price':
                current_pred_price = base_price * pred_val
            else:
                current_pred_price = current_pred_price * (1.0 + pred_val)

            all_x.append(idx[true_index])
            all_true.append(true_price)
            all_pred.append(current_pred_price)

    if not all_x:
        return fig, 0.0, 0.0, 0

    true_sorted = np.array(all_true)
    pred_sorted = np.array(all_pred)

    fig.add_trace(go.Scatter(
        x=all_x, y=pred_sorted,
        mode='markers', name='Prix prédit (généralisation)',
        marker={'size': 5, 'color': '#00E0FF', 'symbol': 'diamond'}
    ))

    errors = pred_sorted - true_sorted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    n_points = int(len(errors))

    return fig, mae, rmse, n_points


def build_training_history_figure(
    losses: List[float],
    val_losses: List[float],
    accs: List[float],
    val_accs: List[float]
) -> go.Figure:
    """
    Construit le graphique de l'historique d'entraînement.
    """
    fig = go.Figure()
    
    if losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(losses) + 1)), y=losses,
            mode='lines+markers', name='Loss train',
            line={'color': '#2ca02c', 'width': 2},
            marker={'size': 6}, yaxis='y'
        ))
    
    if val_losses:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)), y=val_losses,
            mode='lines+markers', name='Loss val',
            line={'color': '#d62728', 'width': 2},
            marker={'size': 6}, yaxis='y'
        ))
    
    if accs:
        accs_pct = [a * 100 for a in accs]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(accs_pct) + 1)), y=accs_pct,
            mode='lines+markers', name='DA train %',
            line={'color': '#1f77b4', 'width': 2, 'dash': 'dot'},
            marker={'size': 6}, yaxis='y2'
        ))
    
    if val_accs:
        vaccs_pct = [a * 100 for a in val_accs]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vaccs_pct) + 1)), y=vaccs_pct,
            mode='lines+markers', name='DA val %',
            line={'color': '#ff7f0e', 'width': 2, 'dash': 'dot'},
            marker={'size': 6}, yaxis='y2'
        ))
    
    loss_info = ''
    if losses or val_losses:
        all_loss = [l for l in (losses + val_losses) if l is not None and l > 0]
        if all_loss:
            current_loss = float(all_loss[-1])
            loss_info = f' (actuel: {current_loss:.2e})' if current_loss < 0.001 else f' (actuel: {current_loss:.6f})'
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        font={'color': '#FFFFFF'},
        title=f'📊 Loss{loss_info} & DA',
        height=300,
        uirevision='play_hist',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font={'size': 10}),
        margin=dict(t=60, b=40, l=60, r=60),
        yaxis={'title': 'Loss', 'side': 'left', 'type': 'log'},
        yaxis2={'title': 'DA %', 'overlaying': 'y', 'side': 'right', 'range': [0, 100], 'ticksuffix': '%'}
    )
    
    return fig


def build_trades_table(trades: List[Dict]) -> List[Dict]:
    """
    Construit les données pour un tableau de trades.
    Retourne une liste de dictionnaires (données brutes, pas de HTML).
    """
    if not trades:
        return []
    
    result = []
    for i, t in enumerate(trades[-30:], 1):
        result.append({
            'index': i,
            'direction': t.get('direction', 'LONG'),
            'entry_date': t.get('entry_time', '-')[:10] if len(t.get('entry_time', '-')) >= 10 else '-',
            'entry_hour': t.get('entry_time', '-')[11:16] if len(t.get('entry_time', '-')) >= 16 else '-',
            'exit_hour': t.get('exit_time', '-')[11:16] if len(t.get('exit_time', '-')) >= 16 else '-',
            'qty': t.get('qty', 0),
            'entry_price': round(t.get('entry_price', 0), 4),
            'exit_price': round(t.get('exit_price', 0), 4),
            'pnl': round(t.get('pnl', 0), 2),
        })
    
    return result

