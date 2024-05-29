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