"""
Module de chargement des données de ventes.

Ce module fournit des utilitaires pour charger et valider les données
brutes de ventes des magasins.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Classe pour charger les données de ventes depuis différentes sources.

    Cette classe implémente des méthodes pour charger, valider et
    effectuer une première inspection des données.

    Attributes:
        data_path (Path): Chemin vers le fichier de données
        df (pd.DataFrame): DataFrame contenant les données chargées
    """

    def __init__(self, data_path: str = "artifacts/data/raw/stores_sales_forecasting.csv"):
        """
        Initialise le DataLoader.

        Args:
            data_path: Chemin vers le fichier CSV des données de ventes
            (par défaut: artifacts/data/raw/)
        """
        self.data_path = Path(data_path)
        self.df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV.

        Returns:
            DataFrame pandas contenant les données de ventes

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le fichier est vide ou mal formaté
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.data_path}")

        logger.info(f"Chargement des données depuis {self.data_path}")

        try:
            # Essayer différents encodages pour gérer divers formats de CSV
            encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
            last_error = None

            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(
                        f"Données chargées avec encoding {encoding}: {self.df.shape[0]} lignes, \
                            {self.df.shape[1]} colonnes"
                    )
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
            else:
                # Si tous les encodages échouent, lever la dernière erreur
                raise last_error

            # Validation basique
            if self.df.empty:
                raise ValueError("Le fichier chargé est vide")

            return self.df

        except pd.errors.EmptyDataError:
            raise ValueError("Le fichier CSV est vide")
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {str(e)}")
            raise

    def get_basic_info(self) -> dict:
        """
        Retourne des informations basiques sur les données.

        Returns:
            Dictionnaire contenant les statistiques de base
        """
        if self.df is None:
            raise ValueError("Aucune donnée chargée. Appelez load_data() d'abord.")

        info = {
            "n_lignes": self.df.shape[0],
            "n_colonnes": self.df.shape[1],
            "colonnes": list(self.df.columns),
            "types": self.df.dtypes.to_dict(),
            "valeurs_manquantes": self.df.isnull().sum().to_dict(),
            "memoire_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
        }

        return info
