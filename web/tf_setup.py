import logging
import Models.tf_config as tf_conf

# Setup GPU dès l'import de ce module
tf_conf.setup_tensorflow_gpu()
