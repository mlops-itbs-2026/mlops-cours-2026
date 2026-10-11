# Fichier : packaging/app/__init__.py
"""Paquet ``app`` : importé par uvicorn sous le nom ``app.serve`` (``--app-dir packaging``).

Ne PAS créer de ``packaging/__init__.py`` : le dossier masquerait le paquet PyPI
``packaging`` dont dépendent pip, MLflow, BentoML…
"""
