import pandas as pd
import numpy as np
from typing import Optional, Literal


CurveType = Literal['random_walk', 'trend', 'seasonal', 'lunch_effect', 'sinusoidale']


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
    freq: str = 'T',
    base_price: float = 100.0,
    data_type: CurveType = 'random_walk',
    seed: Optional[int] = None,
    volatility: float = 0.001,
    trend_strength: float = 0.0001,
    seasonality_amplitude: float = 0.01,
    lunch_effect_strength: float = 0.005,
    extra_noise: float = 0.0,
    volume_base: float = 1_000.0,
    volume_noise: float = 0.10,
    sine_period_minutes: int = 360,
) -> pd.DataFrame:
    """
    Génère une série temporelle minute synthétique avec colonnes 'openPrice' et 'volume'.

    - data_type:
        - 'random_walk': marche aléatoire pure (log‑returns)
        - 'trend': marche + légère tendance
        - 'seasonal': marche + saisonnalité intra‑journalière
        - 'lunch_effect': marche + effet pause déjeuner (12:00–14:00)
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

    if data_type == 'sinusoidale':
        # Courbe sinusoïdale pure (amplitude via seasonality_amplitude) + léger bruit multiplicatif
        t = np.arange(n)
        period = max(1, int(sine_period_minutes))
        amp = float(seasonality_amplitude or 0.01)
        sine = np.sin(2.0 * np.pi * (t % period) / float(period))
        price = float(base_price) * (1.0 + amp * sine)
        if volatility and volatility > 0:
            price = price * (1.0 + rng.normal(0.0, float(volatility), size=n))
    else:
        # Log‑returns aléatoires centrés
        log_rets = rng.normal(loc=0.0, scale=max(1e-9, float(volatility)), size=n)
        # Composante tendance (appliquée comme drift en log‑price)
        if data_type in ('trend', 'seasonal', 'lunch_effect') and trend_strength:
            drift = np.linspace(0.0, float(trend_strength) * n, n)
        else:
            drift = np.zeros(n, dtype=float)

        # Accumuler en log‑price puis exponentier pour garantir positivité
        log_price = np.log(max(1e-9, float(base_price))) + np.cumsum(log_rets) + drift
        price = np.exp(log_price)

    # Saisonnière intra‑jour (minute‑of‑day)
    if data_type == 'seasonal' and seasonality_amplitude:
        minutes_of_day = idx.hour * 60 + idx.minute
        # période ~ 1 jour (minutes 1440) centrée
        season = np.sin(2.0 * np.pi * minutes_of_day / 1440.0)
        price = price * (1.0 + float(seasonality_amplitude) * season)

    # Effet déjeuner (12:00–14:00), multiplicatif négatif
    if data_type == 'lunch_effect' and lunch_effect_strength:
        lunch_mask = (idx.hour >= 12) & (idx.hour < 14)
        lunch_factor = np.where(lunch_mask, (1.0 - float(lunch_effect_strength)), 1.0)
        price = price * lunch_factor

    # Bruit multiplicatif additionnel
    if extra_noise and extra_noise > 0:
        noise = rng.normal(loc=0.0, scale=float(extra_noise), size=n)
        price = price * (1.0 + noise)

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


