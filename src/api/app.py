"""
Application FastAPI pour servir le modèle de prévision des ventes.

Cette API propose des endpoints pour :
- Obtenir des prédictions pour des villes et dates spécifiques
- Consulter l'importance des variables
- Réaliser un contrôle de santé

Utilisation :
    uvicorn src.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configurer la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser l'application FastAPI
app = FastAPI(
    title="Sales Forecasting API",
    description="API for predicting daily sales by city",
    version="1.0.0",
)

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stockage global du modèle
MODEL = None
FEATURE_IMPORTANCE = None
MODEL_METADATA = None


class PredictionRequest(BaseModel):
    """Modèle de requête pour les prédictions."""

    city: str = Field(..., description="City name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    dow: int | None = Field(
        None, description="Day of week (0=Mon, 6=Sun). Auto-calculated if not provided."
    )
    month: int | None = Field(None, description="Month (1-12). Auto-calculated if not provided.")
    is_month_end: int | None = Field(
        None, description="1 if last day of month. Auto-calculated if not provided."
    )
    weekofyear: int | None = Field(
        None, description="Week of year. Auto-calculated if not provided."
    )
    lag_1: float | None = Field(None, description="Sales 1 day ago")
    lag_7: float | None = Field(None, description="Sales 7 days ago")
    rollmean_7: float | None = Field(None, description="7-day rolling mean")
    rollmean_28: float | None = Field(None, description="28-day rolling mean")
    rollstd_28: float | None = Field(None, description="28-day rolling std")

    class Config:
        json_schema_extra = {
            "example": {
                "city": "New York",
                "date": "2017-12-31",
                "lag_1": 500.0,
                "lag_7": 450.0,
                "rollmean_7": 475.0,
                "rollmean_28": 520.0,
                "rollstd_28": 80.0,
            }
        }


class PredictionResponse(BaseModel):
    """Modèle de réponse pour les prédictions."""

    city: str
    date: str
    predicted_sales: float
    features_used: dict


class BatchPredictionRequest(BaseModel):
    """Modèle de requête pour des prédictions en lot."""

    predictions: list[PredictionRequest]


class FeatureImportanceResponse(BaseModel):
    """Modèle de réponse pour l'importance des variables."""

    feature: str
    importance: float


def load_model(model_path: str = "artifacts/models/random_forest_model.joblib"):
    """Charger le modèle entraîné ainsi que ses métadonnées."""
    global MODEL, FEATURE_IMPORTANCE, MODEL_METADATA

    try:
        # Charger le modèle
        MODEL = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Charger l'importance des variables
        importance_path = Path(model_path).parent / "feature_importance.csv"
        if importance_path.exists():
            FEATURE_IMPORTANCE = pd.read_csv(importance_path)
            logger.info("Feature importance loaded")

        # Charger les métriques
        metrics_path = Path(model_path).parent / "metrics.json"
        if metrics_path.exists():
            import json

            with open(metrics_path) as f:
                MODEL_METADATA = json.load(f)
            logger.info("Model metadata loaded")

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def extract_calendar_features(date_str: str) -> dict:
    """Extraire les variables calendaires à partir d'une date."""
    date = pd.to_datetime(date_str)
    return {
        "dow": date.dayofweek,
        "month": date.month,
        "is_month_end": 1 if date.is_month_end else 0,
        "weekofyear": date.isocalendar().week,
    }


@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage."""
    success = load_model()
    if not success:
        logger.warning("Model not loaded. Predictions will not be available.")


@app.get("/")
async def root():
    """Endpoint racine."""
    return {
        "message": "Sales Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "feature_importance": "/feature_importance",
            "model_info": "/model_info",
        },
    }


@app.get("/health")
async def health_check():
    """Endpoint de vérification de l'état de santé."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """
    Prédire les ventes pour une ville et une date données.

    Les variables calendaires (dow, month, is_month_end, weekofyear) sont calculées à partir de
    la date si elles ne sont pas fournies. Les variables historiques (lag_*, rollmean_*, rollstd_*)
    doivent être fournies.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extraire ou réutiliser les variables calendaires fournies
        calendar_features = extract_calendar_features(request.date)

        # Construire le dictionnaire de variables
        features = {
            "dow": request.dow if request.dow is not None else calendar_features["dow"],
            "month": request.month if request.month is not None else calendar_features["month"],
            "is_month_end": request.is_month_end
            if request.is_month_end is not None
            else calendar_features["is_month_end"],
            "weekofyear": request.weekofyear
            if request.weekofyear is not None
            else calendar_features["weekofyear"],
            "lag_1": request.lag_1,
            "lag_7": request.lag_7,
            "rollmean_7": request.rollmean_7,
            "rollmean_28": request.rollmean_28,
            "rollstd_28": request.rollstd_28,
        }

        # Vérifier l'absence de variables historiques obligatoires
        if any(
            v is None
            for k, v in features.items()
            if k.startswith(("lag_", "rollmean_", "rollstd_"))
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Historical features (lag_1, lag_7, rollmean_7, "
                    "rollmean_28, rollstd_28) are required"
                ),
            )

        # Créer le DataFrame pour la prédiction
        X = pd.DataFrame([features])  # noqa: N806

        # Effectuer la prédiction
        prediction = MODEL.predict(X)[0]

        return PredictionResponse(
            city=request.city,
            date=request.date,
            predicted_sales=float(prediction),
            features_used=features,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Prédire les ventes pour plusieurs villes et dates."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for pred_request in request.predictions:
        try:
            result = await predict_sales(pred_request)
            results.append(result.dict())
        except Exception as e:
            results.append({"city": pred_request.city, "date": pred_request.date, "error": str(e)})

    return {"predictions": results}


@app.get("/feature_importance", response_model=list[FeatureImportanceResponse])
async def get_feature_importance():
    """Récupérer l'importance des variables du modèle entraîné."""
    if FEATURE_IMPORTANCE is None:
        raise HTTPException(status_code=503, detail="Feature importance not available")

    return [
        FeatureImportanceResponse(feature=row["feature"], importance=row["importance"])
        for _, row in FEATURE_IMPORTANCE.iterrows()
    ]


@app.get("/model_info")
async def get_model_info():
    """Obtenir les métadonnées du modèle et ses métriques de performance."""
    if MODEL_METADATA is None:
        return {
            "model_type": "RandomForestRegressor",
            "status": "loaded",
            "metrics": "not available",
        }

    return {
        "model_type": "RandomForestRegressor",
        "status": "loaded",
        "metrics": MODEL_METADATA,
        "features": [
            "dow",
            "month",
            "is_month_end",
            "weekofyear",
            "lag_1",
            "lag_7",
            "rollmean_7",
            "rollmean_28",
            "rollstd_28",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
