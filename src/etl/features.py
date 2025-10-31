"""
Module d'ingénierie des features pour séries temporelles.

Ce module crée des variables temporelles ainsi que des features de décalage/roulantes
pour la prévision des ventes. Les features sont générées APRÈS le découpage train/test
afin d'éviter le data leakage.

Features créées :
- Calendrier : dow (jour de la semaine), month, is_month_end, weekofyear
- Historique : lag_1, lag_7, rollmean_7, rollmean_28, rollstd_28
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Créer des features pour la prévision des ventes.

    IMPORTANT : les features de décalage et roulantes sont calculées par groupe (City)
    afin d'éviter les fuites de données entre les différentes villes.
    """

    def __init__(self, df: pd.DataFrame, date_col: str = "Date", group_col: str = "City"):
        """
        Initialiser l'ingénieur de features.

        Args:
            df: DataFrame avec City, Date, Sales
            date_col: Nom de la colonne de date
            group_col: Nom de la colonne de regroupement (City)
        """
        self.df = df.copy()
        self.date_col = date_col
        self.group_col = group_col

        # S'assurer que la date est au format datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Trier par groupe et date (crucial pour les lags/roulants)
        self.df = self.df.sort_values([group_col, date_col]).reset_index(drop=True)

        logger.info(f"FeatureEngineer initialized with {len(self.df)} rows")

    def create_temporal_features(self) -> pd.DataFrame:
        """
        Créer les features temporelles basées sur le calendrier.

        Features :
        - dow : jour de la semaine (0=lundi, 6=dimanche)
        - month : mois (1-12)
        - is_month_end : booléen indiquant le dernier jour du mois
        - weekofyear : semaine de l'année (1-53)

        Returns:
            DataFrame avec les features temporelles ajoutées
        """
        logger.info("Creating calendar features")

        date_series = self.df[self.date_col]

        # Features calendaires
        self.df["dow"] = date_series.dt.dayofweek  # 0=Monday, 6=Sunday
        self.df["month"] = date_series.dt.month  # 1-12
        self.df["is_month_end"] = date_series.dt.is_month_end.astype(int)  # 0 or 1
        self.df["weekofyear"] = date_series.dt.isocalendar().week  # 1-53

        logger.info("Calendar features created: dow, month, is_month_end, weekofyear")

        return self.df

    def create_lag_features(self, lag_periods: list[int] = [1, 7]) -> pd.DataFrame:
        """
        Créer des features de décalage (valeurs passées) par groupe (City).

        Args:
            lag_periods: Liste des décalages à créer (par défaut : [1, 7])

        Returns:
            DataFrame avec les features de décalage ajoutées
        """
        logger.info(f"Creating lag features by {self.group_col}: {lag_periods}")

        # Créer les lags par groupe pour éviter les fuites entre villes
        for lag in lag_periods:
            col_name = f"lag_{lag}"
            self.df[col_name] = self.df.groupby(self.group_col)["Sales"].shift(lag)
            logger.info(f"  - {col_name} created")

        logger.info(f"Lag features created: {len(lag_periods)} columns")

        return self.df

    def create_rolling_features(self) -> pd.DataFrame:
        """
        Créer des features roulantes (moyennes/écarts-types mobiles) par groupe (City).

        Features :
        - rollmean_7 : moyenne mobile sur 7 jours
        - rollmean_28 : moyenne mobile sur 28 jours
        - rollstd_28 : écart-type mobile sur 28 jours

        Returns:
            DataFrame avec les features roulantes ajoutées
        """
        logger.info(f"Creating rolling features by {self.group_col}")

        # Définir les features roulantes spécifiques
        rolling_specs = [
            (7, "mean", "rollmean_7"),
            (28, "mean", "rollmean_28"),
            (28, "std", "rollstd_28"),
        ]

        # Créer chaque feature roulante par groupe
        for window, agg_func, col_name in rolling_specs:
            self.df[col_name] = (
                self.df.groupby(self.group_col)["Sales"]
                .rolling(window=window, min_periods=1)
                .agg(agg_func)
                .reset_index(level=0, drop=True)
            )
            logger.info(f"  - {col_name} created")

        logger.info("Rolling features created: rollmean_7, rollmean_28, rollstd_28")

        return self.df

    def create_all_features(self) -> pd.DataFrame:
        """
        Pipeline complet de création de features.

        Returns:
            DataFrame avec l'ensemble des features
        """
        logger.info("=" * 80)
        logger.info("FEATURE CREATION")
        logger.info(f"Initial shape: {self.df.shape}")
        logger.info("=" * 80)

        initial_cols = self.df.shape[1]

        # 1. Calendar features
        self.create_temporal_features()

        # 2. Lag features
        self.create_lag_features(lag_periods=[1, 7])

        # 3. Rolling features
        self.create_rolling_features()

        logger.info("=" * 80)
        logger.info("FEATURES CREATED")
        logger.info(f"Final shape: {self.df.shape}")
        logger.info(f"New features: {self.df.shape[1] - initial_cols}")
        logger.info(f"Columns: {list(self.df.columns)}")
        logger.info("=" * 80)

        return self.df

    def remove_na_rows(self) -> pd.DataFrame:
        """
        Supprimer les lignes contenant des NaN (créés par les lags/roulants).

        Returns:
            DataFrame sans NaN
        """
        n_rows_before = len(self.df)
        self.df = self.df.dropna()
        n_rows_after = len(self.df)

        logger.info(
            f"Removing NaN: {n_rows_before - n_rows_after} rows removed "
            f"({(n_rows_before - n_rows_after)/n_rows_before*100:.1f}%)"
        )

        return self.df

    def save(self, output_path: str):
        """
        Sauvegarder les données enrichies des features.

        Args:
            output_path: Chemin de sauvegarde
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")


def create_features_for_split(input_path: str, output_path: str, remove_na: bool = True):
    """
    Créer les features pour un jeu de données (train ou test).

    Args:
        input_path: Chemin du fichier en entrée (train.csv ou test.csv)
        output_path: Chemin du fichier en sortie
        remove_na: Si True, supprime les lignes contenant des NaN
    """
    logger.info(f"\nProcessing {input_path}")

    # Charger les données
    df = pd.read_csv(input_path)
    logger.info(f"Data loaded: {df.shape}")

    # Créer les features
    engineer = FeatureEngineer(df, date_col="Date", group_col="City")
    engineer.create_all_features()

    # Supprimer les NaN si demandé
    if remove_na:
        engineer.remove_na_rows()

    # Sauvegarder
    engineer.save(output_path)

    logger.info(f"OK: Features created and saved to {output_path}\n")


def main():
    """Script principal pour créer les features des jeux train et test."""
    # Configurer la journalisation
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 80)
    logger.info("CREATING FEATURES FOR TRAIN AND TEST")
    logger.info("=" * 80)

    # Créer les features pour le jeu d'entraînement
    create_features_for_split(
        input_path="artifacts/data/processed/train.csv",
        output_path="artifacts/data/processed/train_features.csv",
        remove_na=True,
    )

    # Créer les features pour le jeu de test
    create_features_for_split(
        input_path="artifacts/data/processed/test.csv",
        output_path="artifacts/data/processed/test_features.csv",
        remove_na=True,
    )

    logger.info("=" * 80)
    logger.info("FEATURE CREATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
