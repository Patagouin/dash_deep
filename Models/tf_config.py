"""
Centralisation de la configuration TensorFlow pour éviter les problèmes d'initialisation CUDA.
Ce module doit être importé avant toute autre utilisation de TensorFlow.
"""
import os
import logging

# Logger spécifique
logger = logging.getLogger("TF_CONFIG")

def setup_tensorflow_gpu():
    """
    Configure TensorFlow pour utiliser le GPU avec croissance mémoire.
    À appeler au démarrage de l'application.
    """
    # Réduire le verbiage TensorFlow (0 = tous, 1 = info, 2 = warning, 3 = error)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    
    try:
        import tensorflow as tf
        
        # Lister les périphériques physiques
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU(s) détecté(s): {len(gpus)}")
            for gpu in gpus:
                try:
                    # Activer la croissance progressive de la mémoire
                    # C'est CRITIQUE pour les cartes avec peu de VRAM (comme 3GB)
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Memory growth activé pour {gpu.name}")
                except RuntimeError as e:
                    # La croissance de la mémoire doit être définie avant l'initialisation des GPU
                    # Si TF est déjà initialisé, c'est trop tard, mais on log juste un warning
                    logger.warning(f"Impossible de définir memory growth pour {gpu.name} (déjà initialisé?): {e}")
                except Exception as e:
                    logger.error(f"Erreur inattendue lors de la config GPU {gpu.name}: {e}")
        else:
            logger.warning("Aucun GPU détecté par TensorFlow.")

    except ImportError:
        logger.error("TensorFlow n'est pas installé ou ne peut pas être importé.")
    except Exception as e:
        logger.error(f"Erreur globale lors du setup TensorFlow: {e}")

def get_gpu_status():
    """
    Retourne le statut actuel du GPU pour l'affichage UI.
    Ne tente PAS de modifier la configuration.
    
    Returns:
        tuple: (available: bool, message: str)
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # On vérifie si memory_growth est activé (juste pour info)
            try:
                growth = tf.config.experimental.get_memory_growth(gpus[0])
                growth_msg = " (Dynamic Mem)" if growth else ""
            except:
                growth_msg = ""
                
            return True, f"GPU actif: {gpus[0].name}{growth_msg}"
        else:
            return False, "CPU uniquement (Aucun GPU détecté)"
    except Exception as e:
        return False, f"Erreur détection GPU: {str(e)}"

# Alias pour compatibilité
setup_gpu = setup_tensorflow_gpu


