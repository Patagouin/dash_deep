import os

# Réduire le verbiage TensorFlow (0 = tous, 1 = info, 2 = warning, 3 = error)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

try:
    import tensorflow as tf
    # Activer la croissance progressive de la mémoire sur tous les GPU visibles
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
except Exception:
    # S'il y a le moindre souci d'import, on n'empêche pas l'app de démarrer
    pass


