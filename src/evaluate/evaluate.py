"""
Script d'évaluation pour le modèle de prévision des ventes entraîné.

Ce script :
1. Charge le modèle entraîné
2. Charge le jeu de test
3. Produit des prédictions
4. Calcule des métriques complètes
5. Génère des visualisations et des rapports d'évaluation

Utilisation :
    python -m src.evaluate.evaluate
    python -m src.evaluate.evaluate --model-path artifacts/models/my_model.joblib
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import optionnel de seaborn
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurer la journalisation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Charger le modèle entraîné depuis le disque.

    Args:
        model_path: Chemin vers le modèle sauvegardé

    Returns:
        Modèle chargé
    """
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model


def load_test_data(test_path: str) -> pd.DataFrame:
    """
    Charger le jeu de test.

    Args:
        test_path: Chemin vers le CSV de test

    Returns:
        DataFrame de test
    """
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    logger.info(f"Test data loaded: {test_df.shape}")
    return test_df


def prepare_test_features(test_df: pd.DataFrame, target_col: str = "Sales") -> tuple:
    """
    Préparer les features et la cible du jeu de test.

    Args:
        test_df: DataFrame de test
        target_col: Nom de la colonne cible

    Returns:
        Tuple (X_test, y_test, metadata_df)
    """
    logger.info("Preparing test features...")

    # Séparer les métadonnées (City, Date) des features
    metadata_cols = ["City", "Date"]
    metadata_df = test_df[metadata_cols].copy()

    # Retirer les colonnes qui ne sont pas des features
    cols_to_drop = metadata_cols + [target_col]
    X_test = test_df.drop(columns=cols_to_drop)  # noqa: N806
    y_test = test_df[target_col]

    logger.info(f"Features: {list(X_test.columns)}")
    logger.info(f"Test samples: {len(X_test)}")

    return X_test, y_test, metadata_df


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Calculer un ensemble complet de métriques d'évaluation.

    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites

    Returns:
        Dictionnaire de métriques
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
        "max_error": float(np.max(np.abs(y_true - y_pred))),
        "mean_error": float(np.mean(y_true - y_pred)),
        "n_samples": len(y_true),
    }

    return metrics


def display_metrics(metrics: dict):
    """
    Afficher les métriques d'évaluation de manière formatée.

    Args:
        metrics: Dictionnaire de métriques
    """
    logger.info("=" * 100)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 100)

    logger.info("\nRegression Metrics:")
    logger.info(f"  MAE (Mean Absolute Error):     ${metrics['mae']:>10,.2f}")
    logger.info(f"  RMSE (Root Mean Squared Error): ${metrics['rmse']:>10,.2f}")
    logger.info(f"  R² Score:                       {metrics['r2']:>11.4f}")
    logger.info(f"  MAPE (Mean Absolute % Error):   {metrics['mape']:>10.2f}%")

    logger.info("\nError Statistics:")
    logger.info(f"  Median Absolute Error:          ${metrics['median_abs_error']:>10,.2f}")
    logger.info(f"  Max Error:                      ${metrics['max_error']:>10,.2f}")
    logger.info(f"  Mean Error (Bias):              ${metrics['mean_error']:>10,.2f}")

    logger.info("\nDataset Info:")
    logger.info(f"  Number of samples:              {metrics['n_samples']:>11,}")

    logger.info("=" * 100)


def create_predictions_dataframe(
    y_true: pd.Series, y_pred: np.ndarray, metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Créer un DataFrame contenant prédictions et métadonnées.

    Args:
        y_true: Valeurs réelles
        y_pred: Valeurs prédites
        metadata_df: Métadonnées (City, Date)

    Returns:
        DataFrame avec les prédictions
    """
    predictions_df = metadata_df.copy()
    predictions_df["y_true"] = y_true.values
    predictions_df["y_pred"] = y_pred
    predictions_df["error"] = y_true.values - y_pred
    predictions_df["abs_error"] = np.abs(predictions_df["error"])
    predictions_df["pct_error"] = (predictions_df["error"] / y_true.values) * 100

    return predictions_df


def generate_visualizations(predictions_df: pd.DataFrame, output_dir: str):
    """
    Générer les visualisations d'évaluation.

    Args:
        predictions_df: DataFrame avec les prédictions
        output_dir: Répertoire de sauvegarde des graphiques
    """
    logger.info("\n" + "=" * 100)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 100)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Définir le style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Réel vs Prédit
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(predictions_df["y_true"], predictions_df["y_pred"], alpha=0.6, s=50)

    # Ligne de prédiction parfaite
    min_val = min(predictions_df["y_true"].min(), predictions_df["y_pred"].min())
    max_val = max(predictions_df["y_true"].max(), predictions_df["y_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    ax.set_xlabel("Actual Sales ($)", fontsize=12)
    ax.set_ylabel("Predicted Sales ($)", fontsize=12)
    ax.set_title("Actual vs Predicted Sales", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {plot_path}")

    # 2. Graphique des résidus
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(predictions_df["y_pred"], predictions_df["error"], alpha=0.6, s=50)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Sales ($)", fontsize=12)
    ax.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    ax.set_title("Residual Plot", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "residuals_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {plot_path}")

    # 3. Distribution des erreurs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme des erreurs absolues
    axes[0].hist(predictions_df["abs_error"], bins=30, edgecolor="black", alpha=0.7)
    axes[0].axvline(
        predictions_df["abs_error"].median(),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f'Median: ${predictions_df["abs_error"].median():.2f}',
    )
    axes[0].set_xlabel("Absolute Error ($)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of Absolute Errors", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Histogramme des erreurs signées
    axes[1].hist(predictions_df["error"], bins=30, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="r", linestyle="--", linewidth=2, label="Zero error")
    axes[1].set_xlabel("Error (Actual - Predicted)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution of Errors", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plot_path = output_dir / "error_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  ✓ Saved: {plot_path}")

    # 4. Graphique temporel (si Date disponible)
    if "Date" in predictions_df.columns:
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"])
        predictions_df_sorted = predictions_df.sort_values("Date")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            predictions_df_sorted["Date"],
            predictions_df_sorted["y_true"],
            "o-",
            label="Actual",
            alpha=0.7,
            markersize=4,
        )
        ax.plot(
            predictions_df_sorted["Date"],
            predictions_df_sorted["y_pred"],
            "s-",
            label="Predicted",
            alpha=0.7,
            markersize=4,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Sales ($)", fontsize=12)
        ax.set_title("Actual vs Predicted Sales Over Time", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plot_path = output_dir / "time_series_comparison.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Saved: {plot_path}")

    # 5. Erreur par ville (si City disponible)
    if "City" in predictions_df.columns:
        city_errors = (
            predictions_df.groupby("City")["abs_error"].agg(["mean", "count"]).reset_index()
        )
        city_errors = city_errors.sort_values("mean", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(city_errors["City"], city_errors["mean"])
        ax.set_xlabel("Mean Absolute Error ($)", fontsize=12)
        ax.set_ylabel("City", fontsize=12)
        ax.set_title("Top 10 Cities by Mean Absolute Error", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Ajouter des étiquettes de compte
        for i, (idx, row) in enumerate(city_errors.iterrows()):
            ax.text(row["mean"], i, f" n={int(row['count'])}", va="center", fontsize=9)

        plot_path = output_dir / "error_by_city.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  ✓ Saved: {plot_path}")

    logger.info("=" * 100)


def save_evaluation_report(metrics: dict, predictions_df: pd.DataFrame, output_dir: str):
    """
    Sauvegarder le rapport d'évaluation et les prédictions.

    Args:
        metrics: Métriques d'évaluation
        predictions_df: DataFrame contenant les prédictions
        output_dir: Répertoire de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder les métriques au format JSON
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_path}")

    # Sauvegarder les prédictions en CSV
    predictions_path = output_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")

    # Sauvegarder un rapport synthétique
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Regression Metrics:\n")
        f.write(f"  MAE:  ${metrics['mae']:,.2f}\n")
        f.write(f"  RMSE: ${metrics['rmse']:,.2f}\n")
        f.write(f"  R²:   {metrics['r2']:.4f}\n")
        f.write(f"  MAPE: {metrics['mape']:.2f}%\n\n")

        f.write("Error Statistics:\n")
        f.write(f"  Median Absolute Error: ${metrics['median_abs_error']:,.2f}\n")
        f.write(f"  Max Error:             ${metrics['max_error']:,.2f}\n")
        f.write(f"  Mean Error (Bias):     ${metrics['mean_error']:,.2f}\n\n")

        f.write(f"Dataset: {metrics['n_samples']} test samples\n")
        f.write("=" * 100 + "\n")

    logger.info(f"Evaluation report saved to: {report_path}")


def main():
    """Pipeline principal d'évaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Random Forest model on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/random_forest_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="artifacts/data/split/test_features.csv",
        help="Path to test data with features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    logger.info("=" * 100)
    logger.info("MODEL EVALUATION PIPELINE")
    logger.info("=" * 100)

    # 1. Charger le modèle
    model = load_model(args.model_path)

    # 2. Charger les données de test
    test_df = load_test_data(args.test_data)

    # 3. Préparer les features
    X_test, y_test, metadata_df = prepare_test_features(test_df)  # noqa: N806

    # 4. Produire les prédictions
    logger.info("\nMaking predictions...")
    y_pred = model.predict(X_test)
    logger.info(f"Predictions completed for {len(y_pred)} samples")

    # 5. Calculer les métriques
    metrics = calculate_metrics(y_test, y_pred)
    display_metrics(metrics)

    # 6. Créer le DataFrame de prédictions
    predictions_df = create_predictions_dataframe(y_test, y_pred, metadata_df)

    # 7. Générer les visualisations
    generate_visualizations(predictions_df, args.output_dir)

    # 8. Sauvegarder le rapport
    save_evaluation_report(metrics, predictions_df, args.output_dir)

    logger.info("\n" + "=" * 100)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 100)
    logger.info("\nKey Metrics:")
    logger.info(f"  Test R²:  {metrics['r2']:.4f}")
    logger.info(f"  Test MAE: ${metrics['mae']:,.2f}")
    logger.info(f"  Test RMSE: ${metrics['rmse']:,.2f}")
    logger.info(f"\nOutputs saved to: {args.output_dir}")
    logger.info("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
