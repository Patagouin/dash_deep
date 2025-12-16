"""
DiskcacheManager personnalisé qui utilise 'spawn' au lieu de 'fork' pour support GPU.
Cela évite d'avoir besoin de Celery/Redis.
"""
import multiprocessing
import diskcache
from dash.background_callback.managers.diskcache_manager import DiskcacheManager as BaseDiskcacheManager


class SpawnDiskcacheManager(BaseDiskcacheManager):
    """
    DiskcacheManager qui utilise 'spawn' au lieu de 'fork' pour créer les workers.
    Cela permet d'utiliser CUDA dans les background callbacks sans avoir besoin de Celery.
    
    IMPORTANT: Le réglage 'spawn' doit être fait AVANT l'import de diskcache ou multiprocess.
    C'est pourquoi on le fait dans index.py au tout début.
    """
    
    def __init__(self, cache=None, cache_by=None, expire=None):
        # Vérifier que 'spawn' est bien configuré
        # (normalement déjà fait dans index.py, mais on vérifie)
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # Déjà défini ou erreur, on continue
            pass
        
        # Initialiser le cache si non fourni
        if cache is None:
            cache = diskcache.Cache("./cache")
        
        # Appeler le parent
        super().__init__(cache=cache, cache_by=cache_by, expire=expire)

