"""
Pipeline ETL pour la prévision des ventes.

Ce module fournit un pipeline ETL complet :
1. Load : charger les données brutes depuis un CSV
2. Transform : nettoyer et agréger par City + Date
3. Split : réaliser un découpage temporel train/test
4. Features : créer des variables temporelles, de décalage et roulantes
"""

from src.etl.features import FeatureEngineer, create_features_for_split
from src.etl.load import DataLoader
from src.etl.split import DataSplitter
from src.etl.transform import DataTransformer

__all__ = [
    "DataLoader",
    "DataTransformer",
    "DataSplitter",
    "FeatureEngineer",
    "create_features_for_split",
]
