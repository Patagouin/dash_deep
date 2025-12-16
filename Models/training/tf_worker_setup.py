"""
Module pour configurer CUDA dans les workers spawn avant l'import de TensorFlow.
Ce module doit être importé AVANT tout autre module qui utilise TensorFlow dans un worker.
"""
import os


def setup_cuda_for_worker(use_gpu=True):
    """
    Configure CUDA_VISIBLE_DEVICES AVANT l'import de TensorFlow.
    Cette fonction doit être appelée au tout début d'un worker spawn.
    
    IMPORTANT: Dans un worker spawn, les modules sont réimportés depuis le début,
    donc TensorFlow n'a pas encore été importé. On configure juste CUDA_VISIBLE_DEVICES.
    
    Args:
        use_gpu: Si True, active GPU. Si False, force CPU.
    """
    if not use_gpu:
        # Forcer CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        # Réinitialiser CUDA pour GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # NE PAS supprimer les modules TensorFlow du cache !
    # Cela cause PyExceptionRegistry::Init() already called
    # Dans un worker spawn, les modules sont déjà réimportés proprement

