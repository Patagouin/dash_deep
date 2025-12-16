import os
import sys
from dotenv import load_dotenv
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.SqlCom import SqlCom

# Load env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

def run():
    # Variables d'environnement basées sur web/apps/config.py
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME")

    if not user or not database:
        logger.error(f"Variables d'environnement manquantes dans {env_path}. Vérifiez DB_USER et DB_NAME.")
        # Fallback si l'utilisateur utilise les noms POSTGRES_ (standard docker)
        user = user or os.getenv("POSTGRES_USER", "postgres")
        password = password or os.getenv("POSTGRES_PASSWORD", "password")
        database = database or os.getenv("POSTGRES_DB", "postgres")
        host = host or os.getenv("POSTGRES_HOST", "localhost")
        port = port or os.getenv("POSTGRES_PORT", "5432")
        logger.info(f"Tentative avec fallback: {user}@{host}/{database}")

    logger.info(f"Connexion à la base {database} sur {host} avec user {user}...")
    sql = None
    try:
        sql = SqlCom(user, password, host, port, database, None)
    except Exception as e:
        logger.error(f"Échec de la connexion: {e}")
        return

    migration_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'DB', 'migration_transformer_hybrid.sql')
    
    if not os.path.exists(migration_file):
        logger.error(f"Fichier de migration introuvable: {migration_file}")
        return

    logger.info(f"Lecture du fichier de migration: {migration_file}")
    with open(migration_file, 'r') as f:
        content = f.read()

    logger.info("Exécution de la migration...")
    try:
        # Exécution simple du fichier SQL
        sql.cursor.execute(content)
        sql.connection.commit()
        logger.info("Migration réussie ! Colonne 'model_type' ajoutée.")
    except Exception as e:
        logger.error(f"Erreur lors de la migration: {e}")
        try:
            sql.connection.rollback()
        except:
            pass

if __name__ == "__main__":
    run()
