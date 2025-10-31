"""
Module de transformation des données sans data leakage.

Ce module transforme les données brutes en format agrégé City + Date
et applique uniquement des transformations sans utiliser de statistiques globales
pour éviter le data leakage.

Transformations appliquées:
- Conversion des types de données
- Correction des anomalies temporelles (Ship Date < Order Date)
- Agrégation par City + Date (somme des ventes)
- Capping des outliers avec des seuils fixes (pas de calcul sur le dataset complet)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Transforme les données brutes sans data leakage.

    Cette classe prépare les données pour le split train/test en:
    1. Nettoyant les anomalies évidentes (dates négatives)
    2. Agrégeant par City + Date
    3. Appliquant un capping des outliers avec seuils fixes
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialise le transformateur.

        Args:
            df: DataFrame brut chargé
        """
        self.df = df.copy()
        self.original_shape = self.df.shape
        logger.info(f"DataTransformer initialisé avec {self.original_shape[0]} lignes")

    def convert_types(self) -> pd.DataFrame:
        """
        Convertit les colonnes vers les types appropriés.

        Returns:
            DataFrame avec types convertis
        """
        logger.info("Conversion des types de données")

        # Dates
        date_cols = ["Order Date", "Ship Date"]
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

        # Numériques
        numeric_cols = ["Sales", "Quantity", "Discount", "Profit"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        logger.info("Types convertis avec succès")
        return self.df

    def fix_temporal_anomalies(self) -> pd.DataFrame:
        """
        Corrige les anomalies temporelles simples (Ship Date < Order Date).

        Cette correction ne cause pas de data leakage car elle ne dépend
        que de la cohérence interne de chaque ligne.

        Returns:
            DataFrame avec dates corrigées
        """
        logger.info("Correction des anomalies temporelles")

        # Identifier les dates négatives
        negative_mask = self.df["Ship Date"] < self.df["Order Date"]
        n_negative = negative_mask.sum()

        if n_negative > 0:
            logger.warning(f"Correction de {n_negative} dates de livraison négatives")
            # Fixer Ship Date = Order Date + 1 jour pour ces cas
            self.df.loc[negative_mask, "Ship Date"] = self.df.loc[
                negative_mask, "Order Date"
            ] + pd.Timedelta(days=1)

        logger.info("Anomalies temporelles corrigées")
        return self.df

    def aggregate_by_city_date(self) -> pd.DataFrame:
        """
        Agrège les données par City + Order Date.

        Pour chaque combinaison City + Date, on somme les ventes.

        Returns:
            DataFrame agrégé avec City, Date, et Sales totales
        """
        logger.info("Agrégation par City + Order Date")

        # Assurer que Order Date est à la granularité jour
        self.df["Date"] = self.df["Order Date"].dt.date

        # Agréger: somme des ventes par City + Date
        df_agg = self.df.groupby(["City", "Date"], as_index=False).agg({"Sales": "sum"})

        # Reconvertir Date en datetime pour les features temporelles
        df_agg["Date"] = pd.to_datetime(df_agg["Date"])

        logger.info(f"Données agrégées: {len(df_agg)} lignes (City x Date uniques)")

        self.df = df_agg
        return self.df

    def cap_outliers_fixed_threshold(self, sales_max: float = 10000.0) -> pd.DataFrame:
        """
        Plafonne les outliers avec un seuil fixe (pas de calcul statistique).

        Cette approche évite le data leakage car le seuil est fixé à l'avance,
        pas calculé sur les données.

        Args:
            sales_max: Seuil maximum pour Sales (valeur fixe par domaine)

        Returns:
            DataFrame avec outliers plafonnés
        """
        logger.info(f"Capping des outliers avec seuil fixe: Sales max = ${sales_max:,.0f}")

        # Compter les outliers avant capping
        n_outliers = (self.df["Sales"] > sales_max).sum()

        if n_outliers > 0:
            logger.info(f"Capping de {n_outliers} valeurs au-dessus de ${sales_max:,.0f}")
            self.df["Sales"] = self.df["Sales"].clip(upper=sales_max)

        logger.info("Capping des outliers terminé")
        return self.df

    def transform(self, sales_max: float = 10000.0) -> pd.DataFrame:
        """
        Pipeline complet de transformation sans data leakage.

        Args:
            sales_max: Seuil fixe pour capping des Sales

        Returns:
            DataFrame transformé et agrégé
        """
        logger.info("=" * 80)
        logger.info("DÉBUT DU PIPELINE DE TRANSFORMATION (SANS DATA LEAKAGE)")
        logger.info(f"Shape initiale: {self.original_shape}")
        logger.info("=" * 80)

        # 1. Conversion des types
        self.convert_types()

        # 2. Correction des anomalies temporelles
        self.fix_temporal_anomalies()

        # 3. Agrégation City + Date
        self.aggregate_by_city_date()

        # 4. Capping des outliers avec seuil fixe
        self.cap_outliers_fixed_threshold(sales_max=sales_max)

        # 5. Trier par date (important pour le split temporel)
        self.df = self.df.sort_values("Date").reset_index(drop=True)

        logger.info("=" * 80)
        logger.info("FIN DU PIPELINE DE TRANSFORMATION")
        logger.info(f"Shape finale: {self.df.shape}")
        logger.info(f"Colonnes: {list(self.df.columns)}")
        logger.info(f"Période: {self.df['Date'].min()} à {self.df['Date'].max()}")
        logger.info("=" * 80)

        return self.df

    def save(self, output_path: str):
        """
        Sauvegarde les données transformées.

        Args:
            output_path: Chemin de sauvegarde
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        logger.info(f"Données transformées sauvegardées dans {output_path}")


def main():
    """Script principal pour transformer les données."""
    from src.etl.load import DataLoader

    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Charger les données brutes
    loader = DataLoader()
    df_raw = loader.load_data()

    # Transformer
    transformer = DataTransformer(df_raw)
    transformer.transform(sales_max=10000.0)

    # Sauvegarder
    transformer.save("artifacts/data/processed/sales_transformed.csv")

    logger.info("Transformation terminée avec succès!")


if __name__ == "__main__":
    main()
