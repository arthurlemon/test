"""
Script principal de pipeline ETL pour la prévision des ventes.

Ce script exécute la chaîne ETL complète :
1. Chargement des données brutes depuis artifacts/data/raw/
2. Transformation : nettoyage, agrégation par City+Date, plafonnement des valeurs aberrantes
3. Découpage : séparation temporelle train/test (80/20)
4. Features : création des variables calendaires et historiques

Utilisation :
    python -m src.etl.run_pipeline
    python -m src.etl.run_pipeline --sales-max 8000 --test-size 0.25
"""

import argparse
import logging
import sys
from pathlib import Path

from src.etl import DataLoader, DataSplitter, DataTransformer, create_features_for_split

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurer la journalisation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_etl_pipeline(
    raw_data_path: str = "artifacts/data/raw/stores_sales_forecasting.csv",
    sales_max: float = 10000.0,
    test_size: float = 0.2,
    output_dir: str = "artifacts/data",
):
    """
    Exécuter l'ensemble du pipeline ETL.

    Args:
        raw_data_path: Chemin vers le CSV brut
        sales_max: Seuil fixe pour plafonner les ventes aberrantes
        test_size: Proportion de données pour le jeu de test (0,0-1,0)
        output_dir: Répertoire de sortie pour les données traitées

    Returns:
        Dictionnaire contenant les résultats du pipeline
    """
    logger.info("=" * 100)
    logger.info("STARTING ETL PIPELINE FOR SALES FORECASTING")
    logger.info("=" * 100)

    output_dir = Path(output_dir)
    processed_dir = output_dir / "processed"
    split_dir = output_dir / "split"

    processed_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ========================================
    # ÉTAPE 1 : CHARGER
    # ========================================
    logger.info("\n" + "=" * 100)
    logger.info("STEP 1: LOAD RAW DATA")
    logger.info("=" * 100)

    loader = DataLoader(raw_data_path)
    df_raw = loader.load_data()

    info = loader.get_basic_info()
    logger.info("\nRaw data loaded:")
    logger.info(f"  - Rows: {info['n_lignes']:,}")
    logger.info(f"  - Columns: {info['n_colonnes']}")
    logger.info(f"  - Memory: {info['memoire_mb']:.2f} MB")

    results["raw_shape"] = df_raw.shape

    # ========================================
    # ÉTAPE 2 : TRANSFORMER
    # ========================================
    logger.info("\n" + "=" * 100)
    logger.info("STEP 2: TRANSFORM DATA (AGGREGATE BY CITY + DATE)")
    logger.info("=" * 100)

    transformer = DataTransformer(df_raw)
    df_transformed = transformer.transform(sales_max=sales_max)

    # Sauvegarder les données transformées
    transformed_path = processed_dir / "sales_transformed.csv"
    transformer.save(str(transformed_path))

    results["transformed_shape"] = df_transformed.shape
    results["transformed_path"] = str(transformed_path)

    # ========================================
    # ÉTAPE 3 : DÉCOUPER
    # ========================================
    logger.info("\n" + "=" * 100)
    logger.info("STEP 3: TEMPORAL TRAIN/TEST SPLIT")
    logger.info("=" * 100)

    splitter = DataSplitter(df_transformed, date_col="Date")
    train_df, test_df = splitter.temporal_train_test_split(
        test_size=test_size,
        shuffle=False,  # IMPORTANT : conserver l'ordre chronologique pour les séries temporelles
    )

    # Sauvegarder les découpages
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"
    splitter.save_splits(train_path=str(train_path), test_path=str(test_path))

    split_info = splitter.get_split_info()
    results["train_shape"] = train_df.shape
    results["test_shape"] = test_df.shape
    results["train_path"] = str(train_path)
    results["test_path"] = str(test_path)
    results["split_info"] = split_info

    # ========================================
    # ÉTAPE 4 : FEATURES
    # ========================================
    logger.info("\n" + "=" * 100)
    logger.info("STEP 4: CREATE FEATURES")
    logger.info("=" * 100)

    # Créer les features pour le jeu d'entraînement
    train_features_path = split_dir / "train_features.csv"
    create_features_for_split(
        input_path=str(train_path), output_path=str(train_features_path), remove_na=True
    )

    # Créer les features pour le jeu de test
    test_features_path = split_dir / "test_features.csv"
    create_features_for_split(
        input_path=str(test_path), output_path=str(test_features_path), remove_na=True
    )

    results["train_features_path"] = str(train_features_path)
    results["test_features_path"] = str(test_features_path)

    # ========================================
    # RÉSUMÉ
    # ========================================
    logger.info("\n" + "=" * 100)
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 100)

    logger.info("\nPipeline Summary:")
    logger.info(f"  1. Raw data: {results['raw_shape']} -> {results['transformed_shape']}")
    logger.info(f"  2. Train/test split: {results['train_shape']} / {results['test_shape']}")
    logger.info("  3. Features created with calendar + historical features")

    logger.info("\nOutput Files:")
    logger.info("  Processed:")
    logger.info(f"    - {results['transformed_path']}")
    logger.info("  Split:")
    logger.info(f"    - {results['train_path']} (no features)")
    logger.info(f"    - {results['test_path']} (no features)")
    logger.info(f"    - {results['train_features_path']} ✓ READY FOR TRAINING")
    logger.info(f"    - {results['test_features_path']} ✓ READY FOR EVALUATION")

    logger.info("\nFeatures created:")
    logger.info("  Calendar: dow, month, is_month_end, weekofyear")
    logger.info("  Historical: lag_1, lag_7, rollmean_7, rollmean_28, rollstd_28")

    logger.info("\n" + "=" * 100)
    logger.info("READY FOR MODEL TRAINING!")
    logger.info("=" * 100)

    return results


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Run ETL pipeline for sales forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--raw-data",
        type=str,
        default="artifacts/data/raw/stores_sales_forecasting.csv",
        help="Path to raw CSV data",
    )
    parser.add_argument(
        "--sales-max", type=float, default=10000.0, help="Fixed threshold for sales outlier capping"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of data for test set (0.0-1.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/data",
        help="Output directory (will create processed/ and split/ subdirs)",
    )

    args = parser.parse_args()

    # Exécuter le pipeline
    run_etl_pipeline(
        raw_data_path=args.raw_data,
        sales_max=args.sales_max,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )

    # Retourner le statut de réussite
    return 0


if __name__ == "__main__":
    sys.exit(main())
