"""
Script d'entraînement pour le modèle de prévision des ventes avec Random Forest.

Ce script :
1. Charge les jeux train/test enrichis de features
2. Prépare X et y (features et cible)
3. Entraîne un modèle Random Forest
4. Évalue sur les jeux d'entraînement et de test
5. Sauvegarde le modèle entraîné et les métriques

Utilisation :
    python -m src.train.train_model
    python -m src.train.train_model --n-estimators 300 --max-depth 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configurer la journalisation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str) -> tuple:
    """
    Charger les jeux d'entraînement et de test.

    Args:
        train_path: Chemin vers train_features.csv
        test_path: Chemin vers test_features.csv

    Returns:
        Tuple (train_df, test_df)
    """
    logger.info("Loading datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Train set loaded: {train_df.shape}")
    logger.info(f"Test set loaded: {test_df.shape}")

    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "Sales"
) -> tuple:
    """
    Préparer X et y pour l'entraînement.

    Args:
        train_df: DataFrame d'entraînement
        test_df: DataFrame de test
        target_col: Nom de la colonne cible

    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing features...")

    # Colonnes à exclure (non utilisées comme features)
    cols_to_drop = ["City", "Date", target_col]

    # Extraire les features et la cible
    X_train = train_df.drop(columns=cols_to_drop)  # noqa: N806
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=cols_to_drop)  # noqa: N806
    y_test = test_df[target_col]

    logger.info(f"Features: {list(X_train.columns)}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    logger.info(f"Target: {target_col}")

    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 4,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Entraîner le modèle Random Forest.

    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        n_estimators: Nombre d'arbres
        max_depth: Profondeur maximale
        min_samples_split: Nombre minimum d'échantillons pour scinder un nœud
        min_samples_leaf: Nombre minimum d'échantillons dans une feuille
        random_state: Graine aléatoire

    Returns:
        RandomForestRegressor entraîné
    """
    logger.info("=" * 100)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 100)

    logger.info("\nHyperparameters:")
    logger.info(f"  - n_estimators: {n_estimators}")
    logger.info(f"  - max_depth: {max_depth}")
    logger.info(f"  - min_samples_split: {min_samples_split}")
    logger.info(f"  - min_samples_leaf: {min_samples_leaf}")
    logger.info(f"  - random_state: {random_state}")

    # Initialiser le modèle
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    # Entraîner
    logger.info(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    logger.info("Training completed!")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    X_test: pd.DataFrame,  # noqa: N803
    y_test: pd.Series,
) -> dict:
    """
    Évaluer le modèle sur les jeux d'entraînement et de test.

    Args:
        model: Modèle entraîné
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_test: Features de test
        y_test: Cible de test

    Returns:
        Dictionnaire des métriques
    """
    logger.info("=" * 100)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 100)

    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Métriques
    metrics = {
        "train": {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "r2": r2_score(y_train, y_train_pred),
            "mape": np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100,
        },
        "test": {
            "mae": mean_absolute_error(y_test, y_test_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "r2": r2_score(y_test, y_test_pred),
            "mape": np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100,
        },
    }

    # Afficher les résultats
    logger.info("\nTrain Set Metrics:")
    logger.info(f"  MAE:  ${metrics['train']['mae']:,.2f}")
    logger.info(f"  RMSE: ${metrics['train']['rmse']:,.2f}")
    logger.info(f"  R²:   {metrics['train']['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['train']['mape']:.2f}%")

    logger.info("\nTest Set Metrics:")
    logger.info(f"  MAE:  ${metrics['test']['mae']:,.2f}")
    logger.info(f"  RMSE: ${metrics['test']['rmse']:,.2f}")
    logger.info(f"  R²:   {metrics['test']['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['test']['mape']:.2f}%")

    # Vérifier le surapprentissage
    r2_gap = metrics["train"]["r2"] - metrics["test"]["r2"]
    logger.info("\nOverfitting Check:")
    logger.info(f"  R² gap (train - test): {r2_gap:.4f}")
    if r2_gap > 0.1:
        logger.warning("  ⚠️  Potential overfitting detected (R² gap > 0.1)")
    else:
        logger.info("  ✓ No significant overfitting")

    logger.info("=" * 100)

    return metrics


def get_feature_importance(
    model: RandomForestRegressor, feature_names: list, top_n: int = 10
) -> pd.DataFrame:
    """
    Obtenir l'importance des variables depuis le modèle entraîné.

    Args:
        model: RandomForestRegressor entraîné
        feature_names: Liste des noms de features
        top_n: Nombre de features à retourner

    Returns:
        DataFrame contenant l'importance des variables
    """
    logger.info("\n" + "=" * 100)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 100)

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info(f"\nTop {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']:<20} {row['importance']:.4f}")

    logger.info("=" * 100)

    return importance_df


def save_model_and_metrics(
    model: RandomForestRegressor,
    metrics: dict,
    feature_importance: pd.DataFrame,
    output_dir: str = "artifacts/models",
):
    """
    Sauvegarder le modèle entraîné et les métriques.

    Args:
        model: Modèle entraîné
        metrics: Métriques d'évaluation
        feature_importance: DataFrame d'importance des features
        output_dir: Répertoire de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle
    model_path = output_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to: {model_path}")

    # Sauvegarder les métriques
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Sauvegarder l'importance des variables
    importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved to: {importance_path}")


def main():
    """Pipeline d'entraînement principal."""
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for sales forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default="artifacts/data/split/train_features.csv",
        help="Path to training data with features",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="artifacts/data/split/test_features.csv",
        help="Path to test data with features",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Number of trees in Random Forest"
    )
    parser.add_argument("--max-depth", type=int, default=15, help="Maximum depth of trees")
    parser.add_argument(
        "--min-samples-split", type=int, default=10, help="Minimum samples required to split a node"
    )
    parser.add_argument(
        "--min-samples-leaf", type=int, default=4, help="Minimum samples required in a leaf node"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models",
        help="Output directory for model and metrics",
    )

    args = parser.parse_args()

    logger.info("=" * 100)
    logger.info("RANDOM FOREST TRAINING PIPELINE")
    logger.info("=" * 100)

    # 1. Charger les données
    train_df, test_df = load_data(args.train_data, args.test_data)

    # 2. Préparer les features
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)  # noqa: N806

    # 3. Entraîner le modèle
    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    # 4. Évaluer
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 5. Importance des variables
    feature_importance = get_feature_importance(model, list(X_train.columns), top_n=10)

    # 6. Tout sauvegarder
    save_model_and_metrics(model, metrics, feature_importance, args.output_dir)

    logger.info("\n" + "=" * 100)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 100)
    logger.info(f"\nTest R²: {metrics['test']['r2']:.4f}")
    logger.info(f"Test MAE: ${metrics['test']['mae']:,.2f}")
    logger.info(f"Test MAPE: {metrics['test']['mape']:.2f}%")
    logger.info("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
