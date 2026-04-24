"""Package commun pour les 4 projets FL (FedAvg/FedProx/FedNova/FedSGD).

Regroupe le code dupliqué (modele, donnees, entrainement, metriques, strategies,
simulation straggler) dans un unique endroit. Chaque projet importe depuis
`fl_common.*` au lieu de dupliquer task.py et metrics_utils.py.
"""
