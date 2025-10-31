"""
Tests for ML models using sklearn directly.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TestModels:
    """Tests for sklearn-based models."""

    @pytest.fixture
    def sample_data(self):
        """Create test data."""
        np.random.seed(42)
        n = 1000

        X = pd.DataFrame(  # noqa: N806
            {
                "feature_1": np.random.uniform(0, 100, n),
                "feature_2": np.random.uniform(0, 10, n),
                "feature_3": np.random.randint(1, 5, n),
            }
        )

        y = pd.Series(X["feature_1"] * 2 + X["feature_2"] * 5 + np.random.normal(0, 10, n))

        return X, y

    def test_random_forest_training(self, sample_data):
        """Test Random Forest training."""
        X, y = sample_data  # noqa: N806

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]

    def test_random_forest_prediction(self, sample_data):
        """Test Random Forest predictions."""
        X, y = sample_data  # noqa: N806

        X_train, X_test = X[:800], X[800:]  # noqa: N806
        y_train = y[:800]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)

    def test_model_evaluation_metrics(self, sample_data):
        """Test model evaluation metrics."""
        X, y = sample_data  # noqa: N806
        X_train, X_test = X[:800], X[800:]  # noqa: N806
        y_train, y_test = y[:800], y[800:]

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        assert mae > 0
        assert rmse > 0
        assert -1 <= r2 <= 1
