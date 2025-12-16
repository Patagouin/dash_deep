# -*- coding: utf-8 -*-
"""
État partagé (best-effort) pour la page Playground.

Note:
- Les callbacks background=True peuvent s'exécuter dans un process séparé (spawn),
  donc les mises à jour ici ne sont pas garanties côté process principal.
- On conserve néanmoins cet état pour compatibilité avec l'ancien `playground.py`.
"""

play_last_model = None
play_last_model_meta = {}
play_last_model_path = None


