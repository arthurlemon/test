"""
Tests for ETL data transformation modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.etl import DataSplitter, DataTransformer


class TestETL:
    """Tests for ETL pipeline components."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data matching expected schema."""
        np.random.seed(42)
        n = 1000

        order_dates = pd.date_range("2024-01-01", periods=n, freq="h")
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]

        df = pd.DataFrame(
            {
                "Order Date": np.random.choice(order_dates, n),
                "City": np.random.choice(cities, n),
                "Store": np.random.randint(1, 11, n),
                "Sales": np.random.uniform(10, 1000, n),
                "Quantity": np.random.randint(1, 50, n),
            }
        )

        df["Ship Date"] = df["Order Date"] + pd.Timedelta(days=1)
        return df

    @pytest.fixture
    def sample_aggregated_data(self):
        """Create sample aggregated data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        cities = ["New York", "Los Angeles", "Chicago"]

        data = []
        for city in cities:
            for date in dates:
                data.append(
                    {
                        "City": city,
                        "Date": date,
                        "Sales": np.random.uniform(100, 10000),
                        "Items": np.random.randint(10, 500),
                    }
                )

        return pd.DataFrame(data)

    def test_data_transformer(self, sample_raw_data):
        """Test DataTransformer aggregation."""
        transformer = DataTransformer(sample_raw_data)
        df_transformed = transformer.transform(sales_max=10000.0)

        assert "City" in df_transformed.columns
        assert "Date" in df_transformed.columns
        assert "Sales" in df_transformed.columns
        assert df_transformed.shape[0] <= sample_raw_data.shape[0]

    def test_data_splitter(self, sample_aggregated_data):
        """Test DataSplitter temporal split."""
        splitter = DataSplitter(sample_aggregated_data, date_col="Date")
        train_df, test_df = splitter.temporal_train_test_split(test_size=0.2, shuffle=False)

        assert train_df.shape[0] > 0
        assert test_df.shape[0] > 0
        assert train_df.shape[0] + test_df.shape[0] == sample_aggregated_data.shape[0]

        train_dates = pd.to_datetime(train_df["Date"])
        test_dates = pd.to_datetime(test_df["Date"])
        assert train_dates.max() <= test_dates.min()

    def test_outlier_capping(self, sample_raw_data):
        """Test that transformer caps outliers correctly."""
        sample_raw_data.loc[0, "Sales"] = 50000

        transformer = DataTransformer(sample_raw_data)
        df_transformed = transformer.transform(sales_max=10000.0)

        assert df_transformed["Sales"].max() <= 10000.0
