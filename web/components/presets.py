# -*- coding: utf-8 -*-
"""
Gestion des presets (configurations sauvegardées).
Réutilisable dans prediction, playground, etc.
"""

import json
import os
import logging


def get_config_path():
    """Retourne le chemin du fichier de configuration."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'apps', 'config_data.json'))


def load_config():
    """Charge la configuration depuis le fichier JSON."""
    config_path = get_config_path()
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_config(config):
    """Sauvegarde la configuration dans le fichier JSON."""
    config_path = get_config_path()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_presets():
    """Retourne la liste des presets disponibles."""
    conf = load_config()
    return conf.get('presets', {})


def get_preset_options():
    """Retourne les options de preset pour un dropdown."""
    presets = get_presets()
    return [{'label': name, 'value': name} for name in presets.keys()]


def load_preset(preset_name):
    """
    Charge un preset par nom.
    
    Args:
        preset_name: Nom du preset
        
    Returns:
        dict avec les valeurs du preset ou None si non trouvé
    """
    presets = get_presets()
    return presets.get(preset_name)


def save_preset(preset_name, values):
    """
    Sauvegarde un preset.
    
    Args:
        preset_name: Nom du preset
        values: dict des valeurs à sauvegarder
    """
    conf = load_config()
    if 'presets' not in conf:
        conf['presets'] = {}
    conf['presets'][preset_name] = values
    save_config(conf)


def delete_preset(preset_name):
    """
    Supprime un preset.
    
    Args:
        preset_name: Nom du preset à supprimer
        
    Returns:
        True si supprimé, False si non trouvé
    """
    conf = load_config()
    presets = conf.get('presets', {})
    if preset_name in presets:
        del presets[preset_name]
        conf['presets'] = presets
        save_config(conf)
        return True
    return False


def get_hyperparameter_options(param_name):
    """
    Retourne les options disponibles pour un hyperparamètre donné.
    
    Args:
        param_name: Nom du paramètre (ex: 'look_back_options', 'stride_options')
        
    Returns:
        list d'options
    """
    conf = load_config()
    return conf.get(param_name, [])


def to_dropdown_options(values):
    """
    Convertit une liste de valeurs en options de dropdown.
    
    Args:
        values: Liste de valeurs
        
    Returns:
        Liste de dicts [{'label': str, 'value': val}, ...]
    """
    return [{'label': str(v), 'value': v} for v in (values or [])]


def first_value_or_none(values):
    """Retourne la première valeur d'une liste ou None."""
    if isinstance(values, list) and len(values) > 0:
        return values[0]
    return None





