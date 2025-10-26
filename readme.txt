Task manually done:
- Installed pgagent
- File pgpass.conf modified to add connection string
- Set up agent for weekly backup

penser à clean before restore



Useful request:
Number of row (approx):
SELECT reltuples::bigint
FROM pg_catalog.pg_class
WHERE relname = 'sharesPricesQuots';

Requete de backup in pgadmin:
powershell -Command "$dateNow = Get-Date -format \"yyyy_MM_dd_HH\h_mm\m\"; &\"C:\Program Files\PostgreSQL\13\bin\pg_dump.exe\" --file \"%DASH_DEEP_DB_PATH%\\auto_backup_stocksprices_$dateNow\" --host \"localhost\" --port \"5432\" --username \"postgres\" --no-password --verbose --format=c --blobs \"stocksprices\" "

Requete d'update journalier in pgadmin:
powershell -Command "conda activate AI_python38; cd \"%DASH_DEEP_PATH%\dash_deep\scripts\"; python \"update_data_launcher.py\" "


L'environnement conda 'financial_analysis' est activé via l'extension de VSCode.


=============================================
Mise à jour de la liste de tickers (filtrage)
=============================================

Script: scripts/refresh_ticker_list.py

Objectif:
- Mettre à jour le fichier de tickers `data/1er_filtrage_yahoo_info_dispo.csv` en:
  - Retirant les tickers sans cotations disponibles en base sur les 2 derniers mois.
  - Ajoutant automatiquement les nouvelles actions appartenant aux catégories:
    - EU1: Euronext large cap
    - US1: mega cap — secteur Technology
    - US2: large cap — secteur Technology

Critères appliqués:
- Cotations récentes: existence de données dans `sharesPricesQuots` sur les 60 derniers jours.
- Détection Europe (Euronext): colonnes `fullExchangeName`/`exchange`/`market` contenant "Euronext", "Paris", "Amsterdam", "Brussels", "Lisbon" et `marketCap` ≥ 10e9.
- Détection US Tech:
  - Secteur: `sector` contient "Technology".
  - Pays/Exchange: `country` == "United States" ou exchange contient "NASDAQ"/"NYSE".
  - US1 Mega cap: `marketCap` ≥ 200e9
  - US2 Large cap: 10e9 ≤ `marketCap` < 200e9

Commande d'exécution (chemins absolus recommandés):
```
/projets/dash_deep/venv/bin/python /projets/dash_deep/scripts/refresh_ticker_list.py /projets/dash_deep/data/1er_filtrage_yahoo_info_dispo.csv
```

Comportement:
- Crée automatiquement une sauvegarde du CSV cible: `.../1er_filtrage_yahoo_info_dispo.csv.bak_YYYYMMDD_HHMMSS`.
- Réécrit le CSV avec la liste nettoyée et enrichie.


=============================================
Application Web — Analyse & Simulation
=============================================

Analyse des corrélations (page Analyse):
- Filtrage heures de marché: chaque symbole est filtré sur ses heures principales (`openMarketTime` → `closeMarketTime`). Défaut: 09:30–16:00. Les paires US/EU ont très peu d'intersection et sont de fait ignorées.
- Matrices de corrélation:
  - Fonctions: `corr_matrix_with_lag`, `corr_matrix_max_0_30` (service `web/services/correlation.py`).
  - Mode “hausses uniquement”: `positive_mode='both_positive'` pour ne corréler que les co-augmentations.
  - Robustesse: min 60 points (≥10% des points communs), winsorisation 1%, corrélation de Spearman.
- Top 10 corrélations (0–30 min): calcul incrémental avec balayage des lags et affichage du lag optimal.
- Heatmap: balayage max 0–30 min, hover = lag optimal.

Simulation (page Simulation):
- Deux onglets (Tabs):
  - Lead-lag (2 actions): déclenchement sur hausse de A sur 30 min, tenue de B pendant `lag_minutes`, fenêtre horaire d'entrée/sortie optionnelle.
  - Fenêtre horaire (1 action): achat tous les jours à `Heure début d'achat` et vente à `Heure fin de vente`.
- UI contextuelle: les champs non pertinents (B, seuil, décalage) disparaissent en mode Fenêtre horaire.
- Résumé enrichi:
  - Trades, Win rate, Win rate (non‑zéro), PnL réalisé, Valeur finale (%), Minutes et Jours couverts.
  - Gain moyen/jour: moyenne de `daily_equity.pct_change()` en %.
  - Perf: Fetch/Align/Backtest + détails internes du backtest.
- Backtests (`web/services/backtest.py`):
  - `backtest_lag_correlation(...)` avec `signal_window_minutes`, `buy_start_time`, `sell_end_time`, et métriques `perf` (prep/loop).
  - `backtest_time_window(...)` pour la stratégie horaire quotidienne.
- Builders UI (`web/services/sim_builders.py`): composants réutilisables pour la figure d’equity, les sorties journalières, la table des trades et le résumé.

Bonnes pratiques / limitations:
- Mélange US/EU: peu de minutes en commun ⇒ corrélations instables; préférer des paires du même marché.
- Données insuffisantes: messages “Données insuffisantes …” ou “Séries trop courtes” peuvent apparaître si <10 points minute.
- Les corrélations élevées ne garantissent pas un edge; vérifier la cohérence signal/exécution (lead‑lag).