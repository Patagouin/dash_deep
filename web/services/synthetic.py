import pandas as pd
import numpy as np
from typing import Optional, Literal


CurveType = Literal['random_walk', 'trend', 'seasonal', 'lunch_effect', 'sinusoidale', 'plateau']


def _parse_hhmm(value: str) -> tuple[int, int]:
    try:
        h, m = str(value).split(':')
        return int(h), int(m)
    except Exception:
        return 9, 30


def _minute_mask_between(index: pd.DatetimeIndex, open_time: str, close_time: str) -> np.ndarray:
    open_h, open_m = _parse_hhmm(open_time)
    close_h, close_m = _parse_hhmm(close_time)
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    minutes_of_day = index.hour * 60 + index.minute
    return (minutes_of_day >= open_minutes) & (minutes_of_day <= close_minutes)


def generate_synthetic_timeseries(
    start_dt: pd.Timestamp | str,
    end_dt: pd.Timestamp | str,
    *,
    market_open: str = '09:30',
    market_close: str = '16:00',
    freq: str = 'min',
    base_price: float = 100.0,
    data_type: CurveType = 'random_walk',
    seed: Optional[int] = None,
    noise: float = 0.0,
    trend_strength: float = 0.0001,
    seasonality_amplitude: float = 0.01,
    lunch_effect_strength: float = 0.005,
    volume_base: float = 1_000.0,
    volume_noise: float = 0.10,
    sine_period_minutes: int = 360,
    nb_plateaux: int = 3,
) -> pd.DataFrame:
    """
    Génère une série temporelle minute synthétique avec colonnes 'openPrice' et 'volume'.

    - data_type:
        - 'random_walk': marche aléatoire pure (log‑returns)
        - 'trend': marche + légère tendance
        - 'seasonal': marche + saisonnalité intra‑journalière
        - 'lunch_effect': marche + effet pause déjeuner (12:00–14:00)
        - 'sinusoidale': oscillation sinusoïdale
        - 'plateau': N plateaux répétitifs (déterministe)
    
    - nb_plateaux: Nombre de plateaux pour data_type='plateau' (défaut: 3)
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.to_datetime(start_dt)
    end_ts = pd.to_datetime(end_dt)
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    # Index minute complet
    full_index = pd.date_range(start=start_ts, end=end_ts, freq=freq)
    # Jours ouvrés (lun‑ven)
    weekday_mask = full_index.dayofweek < 5
    # Heures de marché
    time_mask = _minute_mask_between(full_index, market_open, market_close)
    idx = full_index[weekday_mask & time_mask]
    if len(idx) == 0:
        return pd.DataFrame(columns=['openPrice', 'volume'])

    n = len(idx)

    # Convertir noise en float, 0 si None ou négatif
    noise_val = float(noise) if noise is not None and noise > 0 else 0.0
    
    if data_type == 'plateau':
        # Courbe à N plateaux avec niveaux ALÉATOIRES qui se répètent jour après jour
        
        open_h, open_m = _parse_hhmm(market_open)
        close_h, close_m = _parse_hhmm(market_close)
        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        day_length = close_minutes - open_minutes
        
        # Nombre de plateaux (minimum 2)
        nb_plat = max(2, int(nb_plateaux) if nb_plateaux else 3)
        
        # Amplitude pour calculer les niveaux
        amp = float(seasonality_amplitude) if seasonality_amplitude and seasonality_amplitude > 0 else 0.20
        
        # Générer N niveaux de prix distincts et les MÉLANGER aléatoirement
        levels = []
        for p in range(nb_plat):
            # Répartir les niveaux entre -amp et +amp
            factor = 1.0 + amp * (1.0 - 2.0 * p / (nb_plat - 1))
            levels.append(float(base_price) * factor)
        
        # Mélanger les niveaux aléatoirement (ordre différent chaque génération)
        rng.shuffle(levels)
        
        # Calculer le plateau pour chaque minute
        minutes_of_day = (idx.hour * 60 + idx.minute).values
        minutes_since_open = minutes_of_day - open_minutes
        
        # Calculer les bornes de chaque plateau
        segment_length = day_length / nb_plat
        
        # Assigner les niveaux selon la position dans la journée
        price = np.empty(n, dtype=np.float64)
        for i in range(n):
            mins = int(minutes_since_open[i])
            plateau_idx = min(int(mins / segment_length), nb_plat - 1)
            price[i] = levels[plateau_idx]
            
    elif data_type == 'sinusoidale':
        # Courbe sinusoïdale pure (amplitude via seasonality_amplitude)
        t = np.arange(n)
        period = max(1, int(sine_period_minutes))
        amp = float(seasonality_amplitude) if seasonality_amplitude and seasonality_amplitude > 0 else 0.01
        sine = np.sin(2.0 * np.pi * (t % period) / float(period))
        price = float(base_price) * (1.0 + amp * sine)
    else:
        # random_walk, trend, seasonal, lunch_effect
        # Si bruit = 0 : prix constant (pas de random walk)
        if noise_val > 0:
            log_rets = rng.normal(loc=0.0, scale=noise_val, size=n)
        else:
            log_rets = np.zeros(n, dtype=float)
        
        # Composante tendance (appliquée comme drift en log‑price)
        trend_val = float(trend_strength) if trend_strength and trend_strength != 0 else 0.0
        if trend_val != 0:
            drift = np.linspace(0.0, trend_val * n, n)
        else:
            drift = np.zeros(n, dtype=float)

        # Accumuler en log‑price puis exponentier pour garantir positivité
        log_price = np.log(max(1e-9, float(base_price))) + np.cumsum(log_rets) + drift
        price = np.exp(log_price)

    # Effets additionnels applicables à toutes les courbes (sauf si valeur = 0)
    
    # Saisonnière intra‑jour (minute‑of‑day) - seulement pour 'seasonal'
    seas_amp = float(seasonality_amplitude) if seasonality_amplitude and seasonality_amplitude > 0 else 0.0
    if data_type == 'seasonal' and seas_amp > 0:
        minutes_of_day = idx.hour * 60 + idx.minute
        season = np.sin(2.0 * np.pi * minutes_of_day / 1440.0)
        price = price * (1.0 + seas_amp * season)

    # Effet déjeuner (12:00–14:00), multiplicatif négatif - seulement pour 'lunch_effect'
    lunch_val = float(lunch_effect_strength) if lunch_effect_strength and lunch_effect_strength > 0 else 0.0
    if data_type == 'lunch_effect' and lunch_val > 0:
        lunch_mask = (idx.hour >= 12) & (idx.hour < 14)
        lunch_factor = np.where(lunch_mask, (1.0 - lunch_val), 1.0)
        price = price * lunch_factor

    # Bruit multiplicatif - applicable à TOUTES les courbes si > 0
    if noise_val > 0 and data_type in ('plateau', 'sinusoidale'):
        price = price * (1.0 + rng.normal(0.0, noise_val, size=n))

    # Volume synthétique: base modulée par une sinusoïde intra‑jour + bruit
    minutes_of_day = idx.hour * 60 + idx.minute
    vol_pattern = 0.5 + 0.5 * np.sin(2.0 * np.pi * minutes_of_day / 1440.0)
    vol = float(volume_base) * (1.0 + 0.5 * vol_pattern)
    if volume_noise and volume_noise > 0:
        vol = vol * (1.0 + rng.normal(0.0, float(volume_noise), size=n))
    vol = np.maximum(vol, 0.0)

    df = pd.DataFrame(
        {
            'openPrice': price.astype(float),
            'volume': vol.astype(float),
        },
        index=idx,
    )

    # Nettoyage final
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return df


def estimate_nb_quotes_per_day(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    per_day = df.groupby(df.index.normalize()).size()
    if per_day.empty:
        return 0
    # La plupart des jours devraient avoir la même taille
    return int(per_day.mode().iloc[0])


