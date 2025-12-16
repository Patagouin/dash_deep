import pandas as pd
import numpy as np
from typing import Optional, Literal


CurveType = Literal['random_walk', 'trend', 'seasonal', 'lunch_effect', 'sinusoidale', 'plateau', 'pattern']


# Stockage global des amorces et patterns pour réutilisation
_pattern_storage = {
    'amorces': {},      # {amorce_id: np.array de valeurs}
    'patterns': {},     # {pattern_id: np.array de valeurs}
    'amorce_to_patterns': {},  # {amorce_id: [pattern_ids]}
    'pattern_to_amorces': {},  # {pattern_id: [amorce_ids]}
    'seed': None,       # Seed utilisé pour la génération
    'config': {},       # Configuration (nb_patterns, nb_amorces, etc.)
}


def reset_pattern_storage():
    """Réinitialise le stockage des patterns."""
    global _pattern_storage
    _pattern_storage = {
        'amorces': {},
        'patterns': {},
        'amorce_to_patterns': {},
        'pattern_to_amorces': {},
        'seed': None,
        'config': {},
    }


def _generate_curve_segment(rng: np.random.Generator, length: int, base_price: float, amplitude: float = 0.1) -> np.ndarray:
    """
    Génère un segment de courbe aléatoire.
    
    Args:
        rng: Générateur aléatoire
        length: Nombre de points
        base_price: Prix de base
        amplitude: Amplitude des variations (proportion du prix de base)
    
    Returns:
        np.array avec les valeurs du segment
    """
    # Générer une marche aléatoire lisse avec un brownian bridge
    steps = rng.normal(0, 1, length)
    cumulative = np.cumsum(steps)
    # Normaliser pour avoir une amplitude contrôlée
    if len(cumulative) > 1:
        cumulative = cumulative - cumulative[0]  # Partir de 0
        max_abs = np.max(np.abs(cumulative)) or 1.0
        cumulative = cumulative / max_abs * amplitude * base_price
    else:
        cumulative = np.zeros(length)
    
    return base_price + cumulative


def _initialize_patterns_and_amorces(
    rng: np.random.Generator,
    nb_patterns: int,
    nb_amorces: int,
    patterns_par_amorce: int,
    amorces_par_pattern: int,
    duree_amorce: int,
    duree_pattern: int,
    base_price: float,
    amplitude: float,
) -> None:
    """
    Initialise les amorces et patterns avec leurs liens.
    
    La relation entre amorces et patterns est bidirectionnelle :
    - Chaque amorce est liée à `patterns_par_amorce` patterns
    - Chaque pattern est lié à `amorces_par_pattern` amorces
    """
    global _pattern_storage

    nb_patterns = max(1, int(nb_patterns))
    nb_amorces = max(1, int(nb_amorces))
    patterns_par_amorce = max(1, int(patterns_par_amorce))
    amorces_par_pattern = max(1, int(amorces_par_pattern))

    # Contraintes évidentes:
    # - on ne peut pas avoir plus de patterns différents par amorce que nb_patterns
    # - on ne peut pas avoir plus d'amorces différentes par pattern que nb_amorces
    patterns_par_amorce_effectif = min(patterns_par_amorce, nb_patterns)
    amorces_par_pattern_effectif = min(amorces_par_pattern, nb_amorces)
    
    # Générer les amorces
    for i in range(nb_amorces):
        _pattern_storage['amorces'][i] = _generate_curve_segment(
            rng, duree_amorce, base_price, amplitude
        )
    
    # Générer les patterns
    for i in range(nb_patterns):
        _pattern_storage['patterns'][i] = _generate_curve_segment(
            rng, duree_pattern, base_price, amplitude
        )

    # Construire un graphe biparti avec degrés MAX:
    # - deg(amorce) <= patterns_par_amorce_effectif
    # - deg(pattern) <= amorces_par_pattern_effectif
    #
    # On remplit autant que possible, sans jamais dépasser les maxima.
    _pattern_storage['amorce_to_patterns'] = {a: [] for a in range(nb_amorces)}
    _pattern_storage['pattern_to_amorces'] = {p: [] for p in range(nb_patterns)}

    amorce_slots = {a: patterns_par_amorce_effectif for a in range(nb_amorces)}
    pattern_slots = {p: amorces_par_pattern_effectif for p in range(nb_patterns)}

    amorce_ids = list(range(nb_amorces))
    pattern_ids = list(range(nb_patterns))
    rng.shuffle(amorce_ids)
    rng.shuffle(pattern_ids)

    # Tant qu'il existe au moins une amorce avec des slots ET un pattern avec des slots,
    # on tente d'ajouter une arête (a -> p) non dupliquée.
    progress = True
    while progress:
        progress = False

        # Trier (aléatoirement) pour répartir les liens
        rng.shuffle(amorce_ids)
        rng.shuffle(pattern_ids)

        for a in amorce_ids:
            if amorce_slots[a] <= 0:
                continue

            # candidats patterns: slots dispo ET pas déjà lié à cette amorce
            candidates = [
                p for p in pattern_ids
                if pattern_slots[p] > 0 and p not in _pattern_storage['amorce_to_patterns'][a]
            ]
            if not candidates:
                continue

            p = int(rng.choice(candidates))
            _pattern_storage['amorce_to_patterns'][a].append(p)
            _pattern_storage['pattern_to_amorces'][p].append(a)
            amorce_slots[a] -= 1
            pattern_slots[p] -= 1
            progress = True

        # Si plus aucun ajout possible, on sort.
    
    # Stocker la configuration
    _pattern_storage['config'] = {
        'nb_patterns': nb_patterns,
        'nb_amorces': nb_amorces,
        'patterns_par_amorce': patterns_par_amorce_effectif,
        'amorces_par_pattern': amorces_par_pattern_effectif,
        'patterns_par_amorce_demande': patterns_par_amorce,
        'amorces_par_pattern_demande': amorces_par_pattern,
        'duree_amorce': duree_amorce,
        'duree_pattern': duree_pattern,
    }


def _get_amorce_and_pattern_for_day(day_rng: np.random.Generator) -> tuple:
    """
    Sélectionne une amorce et un pattern compatible pour un jour donné.
    
    Returns:
        (amorce_id, pattern_id)
    """
    # Choisir une amorce au hasard
    nb_amorces = len(_pattern_storage['amorces'])
    amorce_id = day_rng.integers(0, nb_amorces)
    
    # Choisir un pattern parmi ceux liés à cette amorce
    linked_patterns = _pattern_storage['amorce_to_patterns'].get(amorce_id, [])
    if linked_patterns:
        pattern_id = day_rng.choice(linked_patterns)
    else:
        # Fallback si pas de lien (ne devrait pas arriver)
        nb_patterns = len(_pattern_storage['patterns'])
        pattern_id = day_rng.integers(0, nb_patterns)
    
    return amorce_id, pattern_id


def _day_rng_for_pattern_selection(
    *,
    base_rng: np.random.Generator,
    seed: Optional[int],
    day: pd.Timestamp,
) -> np.random.Generator:
    """
    Retourne un RNG dédié à la sélection amorce/pattern pour un jour.

    - Si seed est None : sélection vraiment aléatoire à chaque génération (donc un 2e clic "Générer"
      peut produire une courbe différente), tout en réutilisant le stock d'amorces/patterns.
    - Si seed est défini : sélection déterministe par jour (reproductible).
    """
    if seed is None:
        # Seed aléatoire dérivé de base_rng, différent à chaque appel de generate_synthetic_timeseries()
        day_seed = int(base_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return np.random.default_rng(day_seed)

    # Déterministe: combiner le seed utilisateur avec la date (secondes depuis epoch)
    day_seconds = int(pd.Timestamp(day).value // 10**9)
    return np.random.default_rng(int(seed) + day_seconds)


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
    sine_reset_daily: bool = False,
    sine_phase_shift_deg: float = 0.0,
    # Paramètres pour le type 'pattern'
    nb_patterns: int = 4,
    nb_amorces: int = 4,
    duree_amorce: int = 60,
    patterns_par_amorce: int = 2,
    amorces_par_pattern: int = 2,
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
        - 'pattern': amorce + pattern réutilisables avec relations configurable
    
    - nb_plateaux: Nombre de plateaux pour data_type='plateau' (défaut: 3)
    - sine_reset_daily: Si True, la sinusoïde redémarre à t=0 chaque matin.
    - sine_phase_shift_deg: Décalage de phase appliqué chaque jour (cumulatif, en degrés).
    
    Paramètres pour data_type='pattern':
    - nb_patterns: Nombre de patterns différents disponibles
    - nb_amorces: Nombre d'amorces différentes disponibles
    - duree_amorce: Durée de la phase d'amorce en minutes (début de journée)
    - patterns_par_amorce: Nombre de patterns différents associés à chaque amorce
    - amorces_par_pattern: Nombre d'amorces différentes associées à chaque pattern
    
    Exemple: si patterns_par_amorce=2 et amorces_par_pattern=2, on a des couples 
    d'amorces liés à des couples de patterns (relations N:M).
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
        period = max(1, int(sine_period_minutes))
        amp = float(seasonality_amplitude) if seasonality_amplitude and seasonality_amplitude > 0 else 0.01
        
        if sine_reset_daily:
            # t dépend de l'heure dans la journée (reset chaque matin)
            open_h, open_m = _parse_hhmm(market_open)
            open_minutes_val = open_h * 60 + open_m
            minutes_of_day = idx.hour * 60 + idx.minute
            t = (minutes_of_day - open_minutes_val).values
        else:
            # t continu sur toute la période
            t = np.arange(n)
            
        # Décalage de phase qui croît de jour en jour
        day_ids = pd.factorize(idx.normalize())[0]
        phase_rad = np.deg2rad(day_ids * float(sine_phase_shift_deg))
        sine = np.sin(2.0 * np.pi * (t % period) / float(period) + phase_rad)
        price = float(base_price) * (1.0 + amp * sine)

    elif data_type == 'pattern':
        # Type pattern : amorce + pattern avec relations configurables
        
        global _pattern_storage
        
        open_h, open_m = _parse_hhmm(market_open)
        close_h, close_m = _parse_hhmm(market_close)
        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m
        day_length = close_minutes - open_minutes
        
        # Durée de l'amorce et du pattern
        amorce_len = min(int(duree_amorce), day_length - 1)
        pattern_len = day_length - amorce_len
        
        # Amplitude pour la génération des courbes
        amp = float(seasonality_amplitude) if seasonality_amplitude and seasonality_amplitude > 0 else 0.10
        
        # Vérifier si on doit réinitialiser le stockage (changement de config ou de seed)
        config_changed = (
            _pattern_storage['seed'] != seed or
            _pattern_storage['config'].get('nb_patterns') != nb_patterns or
            _pattern_storage['config'].get('nb_amorces') != nb_amorces or
            _pattern_storage['config'].get('patterns_par_amorce') != patterns_par_amorce or
            _pattern_storage['config'].get('amorces_par_pattern') != amorces_par_pattern or
            _pattern_storage['config'].get('duree_amorce') != amorce_len or
            _pattern_storage['config'].get('duree_pattern') != pattern_len or
            not _pattern_storage['amorces']  # Pas encore initialisé
        )
        
        if config_changed:
            # Utiliser un seed dérivé pour la génération des patterns
            pattern_rng = np.random.default_rng(seed if seed is not None else 42)
            _initialize_patterns_and_amorces(
                pattern_rng,
                nb_patterns=max(1, int(nb_patterns)),
                nb_amorces=max(1, int(nb_amorces)),
                patterns_par_amorce=max(1, min(int(patterns_par_amorce), int(nb_patterns))),
                amorces_par_pattern=max(1, min(int(amorces_par_pattern), int(nb_amorces))),
                duree_amorce=amorce_len,
                duree_pattern=pattern_len,
                base_price=float(base_price),
                amplitude=amp,
            )
            _pattern_storage['seed'] = seed
        
        # Générer les prix jour par jour
        price = np.empty(n, dtype=np.float64)
        
        # Récupérer les dates uniques
        unique_days = pd.Series(idx.normalize()).unique()
        
        for day in unique_days:
            # Masque pour ce jour
            day_mask = idx.normalize() == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) == 0:
                continue
            
            # Obtenir les minutes depuis l'ouverture pour ce jour
            day_idx = idx[day_mask]
            minutes_of_day = (day_idx.hour * 60 + day_idx.minute).values
            minutes_since_open = minutes_of_day - open_minutes
            
            # RNG pour sélectionner amorce/pattern
            day_rng = _day_rng_for_pattern_selection(
                base_rng=rng,
                seed=seed,
                day=pd.Timestamp(day),
            )
            
            # Sélectionner une amorce et un pattern pour ce jour
            amorce_id, pattern_id = _get_amorce_and_pattern_for_day(day_rng)
            
            # Récupérer les données d'amorce et de pattern
            amorce_data = _pattern_storage['amorces'][amorce_id]
            pattern_data = _pattern_storage['patterns'][pattern_id]
            
            # Assigner les prix pour ce jour
            for i, global_idx in enumerate(day_indices):
                mins = int(minutes_since_open[i])
                
                if mins < amorce_len:
                    # Phase d'amorce
                    amorce_idx = min(mins, len(amorce_data) - 1)
                    price[global_idx] = amorce_data[amorce_idx]
                else:
                    # Phase de pattern
                    pattern_mins = mins - amorce_len
                    pattern_idx = min(pattern_mins, len(pattern_data) - 1)
                    # Ajuster le pattern pour qu'il parte du dernier prix de l'amorce
                    last_amorce_price = amorce_data[-1] if len(amorce_data) > 0 else base_price
                    pattern_start = pattern_data[0] if len(pattern_data) > 0 else base_price
                    offset = last_amorce_price - pattern_start
                    price[global_idx] = pattern_data[pattern_idx] + offset

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
    if noise_val > 0 and data_type in ('plateau', 'sinusoidale', 'pattern'):
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


