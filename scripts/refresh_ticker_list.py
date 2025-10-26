#!/usr/bin/env python3
import os
import sys
import shutil
import datetime
from typing import List, Set

# Positionnement du chemin projet pour importer Models.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)

import pandas as pd

import Models.Shares as sm


DEFAULT_REL_TICKER_FILE = "../data/1er_filtrage_yahoo_info_dispo.csv"

# Seuils (approx. standard): Large Cap >= 10B, Mega Cap >= 200B
LARGE_CAP_THRESHOLD = 10_000_000_000
MEGA_CAP_THRESHOLD = 200_000_000_000


def resolve_file_path(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(SCRIPT_DIR, rel_path))


def read_tickers_from_file(file_path: str) -> List[str]:
    tickers: List[str] = []
    if not os.path.exists(file_path):
        return tickers
    with open(file_path, 'r') as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            if t.startswith('#'):
                continue
            tickers.append(t)
    return tickers


def backup_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{file_path}.bak_{dt}"
    shutil.copy2(file_path, dst)
    return dst


def write_tickers_to_file(file_path: str, tickers: List[str]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for t in tickers:
            f.write(f"{t}\n")

def write_removed_tickers_file(original_file: str, removed: List[str]) -> str:
    """Ecrit la liste des tickers supprimés à côté du CSV d'entrée.
    Retourne le chemin du fichier écrit.
    """
    dir_name = os.path.dirname(original_file)
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{base_name}.removed_{dt}.txt"
    out_path = os.path.join(dir_name, out_name)
    os.makedirs(dir_name, exist_ok=True)
    with open(out_path, 'w') as f:
        for t in removed:
            f.write(f"{t}\n")
    return out_path

def write_stale_tickers_file(dir_from_csv: str, stale: List[str], days: int) -> str:
    """Ecrit la liste des tickers sans cotations récentes dans le dossier du CSV.
    Retourne le chemin du fichier écrit.
    """
    dir_name = dir_from_csv
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"stale_tickers_{days}d_{dt}.txt"
    out_path = os.path.join(dir_name, out_name)
    os.makedirs(dir_name, exist_ok=True)
    with open(out_path, 'w') as f:
        for t in stale:
            f.write(f"{t}\n")
    return out_path


def get_share_row_by_symbol(df_shares: pd.DataFrame, symbol: str):
    df = df_shares[df_shares['symbol'] == symbol]
    try:
        return next(df.itertuples(index=False))
    except StopIteration:
        return None


def has_recent_quotes(shares_mgr: sm.Shares, share_row, days: int = 60) -> bool:
    if share_row is None:
        return False
    try:
        date_end = datetime.datetime.now()
        date_begin = date_end - datetime.timedelta(days=days)
        df = shares_mgr.getDfDataRangeFromShare(share_row, date_begin, date_end)
        return df is not None and not df.empty
    except Exception:
        return False


def string_contains(value, needle: str) -> bool:
    if value is None:
        return False
    try:
        return needle.lower() in str(value).lower()
    except Exception:
        return False


def any_contains(values: List, needles: List[str]) -> bool:
    for n in needles:
        for v in values:
            if string_contains(v, n):
                return True
    return False


def safe_number(value):
    try:
        return float(value)
    except Exception:
        return None


def filter_euronext_large_cap(df_shares: pd.DataFrame) -> pd.DataFrame:
    cols = df_shares.columns
    market_cap_col = 'marketCap' if 'marketCap' in cols else None
    exchange_candidates = []
    for c in ['fullExchangeName', 'exchange', 'market']:
        if c in cols:
            exchange_candidates.append(c)

    def is_euronext(row) -> bool:
        vals = [row[c] for c in exchange_candidates]
        return any_contains(vals, ['euronext', 'paris', 'amsterdam', 'brussels', 'lisbon'])

    def is_large_cap(row) -> bool:
        if market_cap_col is None:
            return False
        mc = safe_number(row[market_cap_col])
        return mc is not None and mc >= LARGE_CAP_THRESHOLD

    mask = df_shares.apply(lambda r: is_euronext(r) and is_large_cap(r), axis=1)
    return df_shares[mask]


def filter_us_tech_by_cap(df_shares: pd.DataFrame, min_cap: int, max_cap: int = None) -> pd.DataFrame:
    cols = df_shares.columns
    market_cap_col = 'marketCap' if 'marketCap' in cols else None
    sector_col = 'sector' if 'sector' in cols else None
    country_col = 'country' if 'country' in cols else None
    exch_candidates = []
    for c in ['fullExchangeName', 'exchange', 'market']:
        if c in cols:
            exch_candidates.append(c)

    def is_us(row) -> bool:
        if country_col and string_contains(row[country_col], 'united states'):
            return True
        vals = [row[c] for c in exch_candidates]
        return any_contains(vals, ['nasdaq', 'nyse', 'us_market'])

    def is_tech(row) -> bool:
        if not sector_col:
            return False
        return string_contains(row[sector_col], 'technology')

    def cap_ok(row) -> bool:
        if not market_cap_col:
            return False
        mc = safe_number(row[market_cap_col])
        if mc is None:
            return False
        if max_cap is None:
            return mc >= min_cap
        return (mc >= min_cap) and (mc < max_cap)

    mask = df_shares.apply(lambda r: is_us(r) and is_tech(r) and cap_ok(r), axis=1)
    return df_shares[mask]


def build_updated_ticker_set(shares_mgr: sm.Shares, initial_tickers: List[str]) -> List[str]:
    df = shares_mgr.dfShares

    # Nettoyage initial: ne garder que les tickers présents en DB
    initial_in_db: List[str] = []
    for t in initial_tickers:
        if (df['symbol'] == t).any():
            initial_in_db.append(t)

    # 1) Retirer ceux sans cotation sur 2 mois
    kept: Set[str] = set()
    removed_candidates: Set[str] = set()
    for t in initial_in_db:
        row = get_share_row_by_symbol(df, t)
        if has_recent_quotes(shares_mgr, row, days=60):
            kept.add(t)
        else:
            removed_candidates.add(t)

    # 2) Ajouter EU1 (Euronext large cap)
    eu_df = filter_euronext_large_cap(df)
    for row in eu_df.itertuples(index=False):
        sym = getattr(row, 'symbol', None) or row[df.columns.get_loc('symbol')]
        if sym and sym not in kept:
            if has_recent_quotes(shares_mgr, row, days=60):
                kept.add(sym)

    # 3) Ajouter US1 (mega cap tech)
    us1_df = filter_us_tech_by_cap(df, MEGA_CAP_THRESHOLD, None)
    for row in us1_df.itertuples(index=False):
        sym = getattr(row, 'symbol', None) or row[df.columns.get_loc('symbol')]
        if sym and sym not in kept:
            if has_recent_quotes(shares_mgr, row, days=60):
                kept.add(sym)

    # 4) Ajouter US2 (large cap tech)
    us2_df = filter_us_tech_by_cap(df, LARGE_CAP_THRESHOLD, MEGA_CAP_THRESHOLD)
    for row in us2_df.itertuples(index=False):
        sym = getattr(row, 'symbol', None) or row[df.columns.get_loc('symbol')]
        if sym and sym not in kept:
            if has_recent_quotes(shares_mgr, row, days=60):
                kept.add(sym)

    # Sort final pour stabilité
    return sorted(kept), sorted(removed_candidates)

def list_stale_tickers(shares_mgr: sm.Shares, days: int = 60) -> List[str]:
    """Retourne la liste des symboles présents en BDD sans cotations sur 'days' jours."""
    df = shares_mgr.dfShares
    stale: List[str] = []
    for row in df.itertuples(index=False):
        try:
            if not has_recent_quotes(shares_mgr, row, days=days):
                sym = getattr(row, 'symbol', None)
                if sym:
                    stale.append(sym)
        except Exception:
            # En cas d'erreur ponctuelle, considérer comme non-stale (sécurité)
            pass
    return sorted(list(set(stale)))


def main():
    # Fichier cible
    rel_file = DEFAULT_REL_TICKER_FILE
    if len(sys.argv) > 1:
        rel_file = sys.argv[1]
    ticker_file = resolve_file_path(rel_file)

    # Chargement des actions en DB
    shares_mgr = sm.Shares(readOnlyThosetoUpdate=True)

    # Mode: lister les tickers sans cotations récentes (base de données complète)
    if any(arg.startswith('--list-stale') for arg in sys.argv[1:]):
        # Parse --days=N si fourni
        days = 60
        for arg in sys.argv[1:]:
            if arg.startswith('--days='):
                try:
                    days = int(arg.split('=', 1)[1])
                except Exception:
                    pass
        dir_out = os.path.dirname(ticker_file)
        stale_list = list_stale_tickers(shares_mgr, days=days)
        out_path = write_stale_tickers_file(dir_out, stale_list, days)
        print(f"Liste des tickers sans cotations sur {days} jours écrite: {out_path} ({len(stale_list)} tickers)")
        return

    # Lecture liste actuelle (mode mise à jour du CSV)
    current_list = read_tickers_from_file(ticker_file)

    # Calcul de la nouvelle liste
    updated_list, removed_list = build_updated_ticker_set(shares_mgr, current_list)

    # Backup puis écriture
    backup_path = backup_file(ticker_file)
    if backup_path:
        print(f"Backup du fichier existant: {backup_path}")
    write_tickers_to_file(ticker_file, updated_list)
    print(f"Nouvelle liste écrite: {ticker_file} ({len(updated_list)} tickers)")
    # Ecriture de la liste des tickers supprimés pour revue manuelle
    if len(removed_list) > 0:
        removed_path = write_removed_tickers_file(ticker_file, removed_list)
        print(f"Tickers supprimés listés dans: {removed_path} ({len(removed_list)} tickers)")
    else:
        print("Aucun ticker supprimé.")


if __name__ == "__main__":
    main()


