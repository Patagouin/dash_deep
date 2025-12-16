#!/usr/bin/env python3
"""
Nettoie les tickers delistés / invalides Yahoo en:
- forçant isUpdated=false en base (donc ils ne seront plus dans la liste "à updater")
- retirant ces tickers de certains fichiers de listes (optionnel)

Usage (chemins absolus recommandés):

/projets/dash_deep/venv/bin/python /projets/dash_deep/scripts/clean_delisted_tickers.py \
  --lookback-days=30 \
  --update-files=/projets/dash_deep/data/1er_filtrage_yahoo_info_dispo.csv,/projets/dash_deep/data/lists_stocks_EU_US.data
"""

import os
import sys
import datetime
from typing import List, Set

# Positionnement du chemin projet pour importer Models.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)

import Models.Shares as sm
import Models.utils as ut


def parse_args(argv: List[str]):
    lookback_days = 30
    update_files: List[str] = []
    for a in argv:
        if a.startswith("--lookback-days="):
            try:
                lookback_days = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("--update-files="):
            files_str = a.split("=", 1)[1].strip()
            if files_str:
                update_files = [p.strip() for p in files_str.split(",") if p.strip()]
    return lookback_days, update_files


def backup_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = f"{path}.bak_{dt}"
    import shutil
    shutil.copy2(path, dst)
    return dst


def remove_tickers_from_file(path: str, to_remove: Set[str]) -> int:
    """Retire les tickers (lignes exactes) tout en conservant commentaires/structure."""
    if not os.path.exists(path):
        return 0
    with open(path, "r") as f:
        lines = f.readlines()
    new_lines = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped in to_remove:
            removed += 1
            continue
        new_lines.append(line)
    if removed > 0:
        backup_file(path)
        with open(path, "w") as f:
            f.writelines(new_lines)
    return removed


def main():
    lookback_days, update_files = parse_args(sys.argv[1:])

    sh = sm.Shares(readOnlyThosetoUpdate=True)
    invalid: List[str] = []
    reasons = {}

    nb_total = sh.dfShares.shape[0]
    cpt = 0

    for share in sh.dfShares.itertuples():
        cpt += 1
        sym = getattr(share, "symbol", None)
        if not sym:
            continue
        ok, reason = ut.yahoo_symbol_status(sym, lookback_days=lookback_days)
        if not ok:
            invalid.append(sym)
            reasons[sym] = reason
            try:
                sh.disable_updates_for_share(share, reason=reason)
            except Exception:
                pass
        if cpt % 50 == 0:
            print(f"Progress: {cpt}/{nb_total}...")

    invalid = sorted(list(set(invalid)))
    print("")
    print(f"Tickers invalides/delistés détectés: {len(invalid)}")
    for t in invalid:
        print(f"- {t} ({reasons.get(t)})")

    # Optionnel: nettoyer des fichiers de listes
    if update_files:
        print("")
        for fpath in update_files:
            removed_count = remove_tickers_from_file(fpath, set(invalid))
            print(f"Fichier nettoyé: {fpath} (lignes retirées: {removed_count})")

    # Rapport daté
    try:
        out_dir = os.path.join(PROJECT_DIR, "data")
        os.makedirs(out_dir, exist_ok=True)
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"delisted_detected_{dt}.txt")
        with open(out_path, "w") as f:
            for t in invalid:
                f.write(f"{t} ; {reasons.get(t, '')}\n")
        print("")
        print(f"Rapport écrit: {out_path}")
    except Exception as e:
        print(f"Impossible d'écrire le rapport: {e}")


if __name__ == "__main__":
    main()


