"""
Module de découpage temporel train/test.

Ce module implémente un découpage chronologique afin d'éviter le data leakage.
La séparation respecte l'ordre temporel : apprentissage sur les dates anciennes,
test sur les dates récentes.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Découpage temporel train/test pour des séries temporelles.

    Cette classe effectue une séparation chronologique qui respecte l'ordre du temps
    afin d'éviter toute fuite de données.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "Date"):
        """
        Initialiser le découpeur.

        Args:
            df: DataFrame contenant les données transformées
            date_col: Nom de la colonne de date
        """
        self.df = df.copy()
        self.date_col = date_col
        self.train_df = None
        self.test_df = None

        # Vérifier que la colonne de date existe
        if date_col not in self.df.columns:
            raise ValueError(f"Column {date_col} not found in DataFrame")

        # S'assurer que la colonne de date est de type datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])

        logger.info(f"DataSplitter initialized with {len(self.df)} rows")

    def temporal_train_test_split(
        self, test_size: float = 0.2, shuffle: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Réaliser un découpage temporel train/test.

        Args:
            test_size: Proportion de données pour le test (0,0-1,0)
            shuffle: Si False, respecte l'ordre chronologique
            (recommandé pour les séries temporelles)

        Returns:
            Tuple (train_df, test_df)
        """
        logger.info("=" * 80)
        logger.info("TEMPORAL TRAIN/TEST SPLIT")
        logger.info("=" * 80)

        # Trier par date (important !)
        df_sorted = self.df.sort_values(self.date_col).reset_index(drop=True)

        # Calculer l'indice de séparation
        split_idx = int(len(df_sorted) * (1 - test_size))

        if shuffle:
            # Si shuffle=True, utiliser sklearn (mais déconseillé pour les séries temporelles)
            logger.warning("shuffle=True: Split does not respect chronological order!")
            self.train_df, self.test_df = train_test_split(
                df_sorted, test_size=test_size, shuffle=True, random_state=42
            )
        else:
            # Découpage chronologique : train = dates anciennes, test = dates récentes
            self.train_df = df_sorted.iloc[:split_idx].copy()
            self.test_df = df_sorted.iloc[split_idx:].copy()

        # Journaux informatifs
        logger.info(f"\nFull dataset: {len(df_sorted)} rows")
        logger.info(
            f"Train set: {len(self.train_df)} rows ({len(self.train_df) / len(df_sorted) * 100:.1f}%)"  # noqa E501
        )
        logger.info(
            f"Test set: {len(self.test_df)} rows ({len(self.test_df) / len(df_sorted) * 100:.1f}%)"
        )

        logger.info(
            f"\nTrain period: {self.train_df[self.date_col].min().date()} to "
            f"{self.train_df[self.date_col].max().date()}"
        )
        logger.info(
            f"Test period: {self.test_df[self.date_col].min().date()} to "
            f"{self.test_df[self.date_col].max().date()}"
        )

        # Vérifier le chevauchement temporel (si shuffle=False)
        if not shuffle:
            if self.train_df[self.date_col].max() > self.test_df[self.date_col].min():
                logger.warning("Warning: Temporal overlap detected between train and test!")
            else:
                logger.info("OK: No temporal overlap between train and test")

        logger.info("=" * 80)

        return self.train_df, self.test_df

    def save_splits(
        self,
        train_path: str = "artifacts/data/processed/train.csv",
        test_path: str = "artifacts/data/processed/test.csv",
    ):
        """
        Sauvegarder les jeux d'entraînement et de test.

        Args:
            train_path: Chemin de sauvegarde pour l'entraînement
            test_path: Chemin de sauvegarde pour le test
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("No split performed. Call temporal_train_test_split() first.")

        # Créer les répertoires si nécessaire
        Path(train_path).parent.mkdir(parents=True, exist_ok=True)
        Path(test_path).parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder
        self.train_df.to_csv(train_path, index=False)
        self.test_df.to_csv(test_path, index=False)

        logger.info(f"Train set saved to {train_path}")
        logger.info(f"Test set saved to {test_path}")

    def get_split_info(self) -> dict:
        """
        Retourner des informations sur le découpage.

        Returns:
            Dictionnaire contenant les statistiques du découpage
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("No split performed.")

        return {
            "total_rows": len(self.df),
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
            "train_pct": len(self.train_df) / len(self.df) * 100,
            "test_pct": len(self.test_df) / len(self.df) * 100,
            "train_period_start": self.train_df[self.date_col].min(),
            "train_period_end": self.train_df[self.date_col].max(),
            "test_period_start": self.test_df[self.date_col].min(),
            "test_period_end": self.test_df[self.date_col].max(),
        }


def main():
    """Script principal pour découper les données."""
    # Configurer la journalisation
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Charger les données transformées
    df = pd.read_csv("artifacts/data/processed/sales_transformed.csv")
    logger.info(f"Transformed data loaded: {df.shape}")

    # Créer le découpeur
    splitter = DataSplitter(df, date_col="Date")

    # Effectuer le découpage temporel (80/20)
    train_df, test_df = splitter.temporal_train_test_split(test_size=0.2, shuffle=False)

    # Sauvegarder
    splitter.save_splits(
        train_path="artifacts/data/processed/train.csv",
        test_path="artifacts/data/processed/test.csv",
    )

    # Afficher les informations
    info = splitter.get_split_info()
    logger.info("\nSplit summary:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nSplit completed successfully!")


if __name__ == "__main__":
    main()
