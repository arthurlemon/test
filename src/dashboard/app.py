"""
Dashboard Streamlit pour la prévision des ventes.

Fonctionnalités :
- Simulation temporelle : générer des prévisions pour le lendemain/la semaine/le mois suivant
selon les villes
- Importance des variables : comprendre ce qui influence les prédictions
- Affiche uniquement les villes disposant d'un historique suffisant (28+ jours)

Utilisation :
    streamlit run src/dashboard/app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration de la page
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str = "artifacts/models/random_forest_model.joblib"):
    """Charger le modèle entraîné."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_feature_importance():
    """Charger l'importance des variables."""
    try:
        importance_df = pd.read_csv("artifacts/models/feature_importance.csv")
        return importance_df
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
        return None


@st.cache_data
def load_historical_data():
    """Charger les features d'entraînement pour le contexte historique."""
    try:
        train_df = pd.read_csv("artifacts/data/split/train_features.csv")
        train_df["Date"] = pd.to_datetime(train_df["Date"])
        return train_df
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")
        return None


def extract_calendar_features(date: datetime) -> dict:
    """Extraire les variables calendaires à partir d'une date."""
    return {
        "dow": date.dayofweek,
        "month": date.month,
        "is_month_end": 1 if date == date + pd.offsets.MonthEnd(0) else 0,
        "weekofyear": date.isocalendar().week,
    }


def get_cities_with_sufficient_data(hist_df: pd.DataFrame, min_days: int = 28) -> list:
    """Récupérer les villes disposant d'assez d'historique pour les prédictions."""
    city_counts = hist_df.groupby("City").size()
    return sorted(city_counts[city_counts >= min_days].index.tolist())


def get_historical_features(city: str, date: datetime, hist_df: pd.DataFrame) -> dict:
    """Obtenir les features historiques (lag, rolling) depuis les données passées."""
    # Filtrer les données pour cette ville
    city_data = hist_df[hist_df["City"] == city].copy()
    city_data = city_data.sort_values("Date")

    # Trouver les données les plus récentes avant la date cible
    past_data = city_data[city_data["Date"] < date]

    if len(past_data) == 0:
        return None

    # Récupérer les ventes récentes pour les features de décalage
    recent_sales = past_data.tail(30)["Sales"].values

    features = {}

    # Features de décalage
    if len(recent_sales) >= 1:
        features["lag_1"] = recent_sales[-1]
    if len(recent_sales) >= 7:
        features["lag_7"] = recent_sales[-7]

    # Features roulantes
    if len(recent_sales) >= 7:
        features["rollmean_7"] = recent_sales[-7:].mean()
    if len(recent_sales) >= 28:
        features["rollmean_28"] = recent_sales[-28:].mean()
        features["rollstd_28"] = recent_sales[-28:].std()

    return features


def make_prediction(model, city: str, date: datetime, hist_df: pd.DataFrame) -> tuple:
    """Réaliser une prédiction pour une ville et une date."""
    # Variables calendaires
    calendar_features = extract_calendar_features(date)

    # Variables historiques
    hist_features = get_historical_features(city, date, hist_df)

    if hist_features is None or len(hist_features) < 5:
        return None, "Insufficient historical data for this city"

    # Combiner les features
    features = {**calendar_features, **hist_features}

    # Créer le DataFrame
    X = pd.DataFrame([features])  # noqa: N806

    # Prédire
    prediction = model.predict(X)[0]

    return prediction, features


def main():
    # En-tête
    st.markdown(
        '<div class="main-header">📊 Sales Forecasting Dashboard</div>', unsafe_allow_html=True
    )

    # Charger le modèle et les données
    model = load_model()
    feature_importance = load_feature_importance()
    hist_df = load_historical_data()

    if model is None:
        st.error("❌ Model not loaded. Please ensure the model file exists.")
        return

    # Barre latérale
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select Page", ["⏰ Time Simulation", "📈 Feature Importance"], index=0)

    # Pages
    if page == "⏰ Time Simulation":
        show_time_simulation_page(model, hist_df)
    elif page == "📈 Feature Importance":
        show_feature_importance_page(feature_importance)


def show_feature_importance_page(feature_importance):
    """Afficher la page sur l'importance des variables."""
    st.header("📈 Feature Importance")

    if feature_importance is None:
        st.error("Feature importance data not available")
        return

    # Visualiser
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_importance = feature_importance.sort_values("importance", ascending=True)
    ax.barh(sorted_importance["feature"], sorted_importance["importance"])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    st.pyplot(fig)

    # Informations clés
    st.markdown("### 💡 Insights")
    top_feature = feature_importance.sort_values("importance", ascending=False).iloc[0]
    st.info(
        f"""
    The most important feature is **{top_feature['feature']}** with importance
    **{top_feature['importance']:.4f}**.

    This means this feature contributes the most to the model's predictions.
    """
    )


def show_time_simulation_page(model, hist_df):
    """Afficher la page de simulation temporelle pour les prédictions en lot."""
    st.header("⏰ Time Simulation")

    if hist_df is None:
        st.error("Historical data not available")
        return

    st.markdown(
        """
    Generate predictions for the next N days or weeks across multiple cities.
    This simulates rolling forward in time.
    """
    )

    # Obtenir les villes disposant de suffisamment de données
    cities = get_cities_with_sufficient_data(hist_df)

    if not cities:
        st.error("No cities have sufficient historical data (minimum 28 days required)")
        return

    st.info(f"📊 {len(cities)} cities available with sufficient data for predictions")

    # Paramètres
    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon", ["Next Day", "Next Week (7 days)", "Next 2 Weeks", "Next Month"]
        )

    with col2:
        selected_cities = st.multiselect(
            "Select Cities", cities, default=cities[:3] if len(cities) >= 3 else cities
        )

    with col3:
        max_date = hist_df["Date"].max()
        start_date = st.date_input(
            "Start Date", value=max_date + timedelta(days=1), min_value=max_date
        )

    # Convertir l'horizon en nombre de jours
    horizon_map = {"Next Day": 1, "Next Week (7 days)": 7, "Next 2 Weeks": 14, "Next Month": 30}
    n_days = horizon_map[forecast_horizon]

    if st.button("🚀 Generate Forecast", type="primary"):
        if not selected_cities:
            st.warning("Please select at least one city")
            return

        with st.spinner(f"Generating {n_days}-day forecast for {len(selected_cities)} cities..."):
            # Générer les prédictions
            results = []

            for city in selected_cities:
                for day_offset in range(n_days):
                    pred_date = pd.to_datetime(start_date) + timedelta(days=day_offset)
                    prediction, features = make_prediction(model, city, pred_date, hist_df)

                    if prediction is not None:
                        results.append(
                            {
                                "City": city,
                                "Date": pred_date.date(),
                                "Predicted_Sales": prediction,
                                "dow": features["dow"],
                                "lag_1": features.get("lag_1", None),
                            }
                        )

            if results:
                forecast_df = pd.DataFrame(results)

                # Afficher le récapitulatif
                st.success(f"✅ Generated {len(forecast_df)} predictions!")

                # Métriques
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Predicted Sales", f"${forecast_df['Predicted_Sales'].sum():,.2f}"
                    )
                with col2:
                    st.metric(
                        "Average Daily Sales", f"${forecast_df['Predicted_Sales'].mean():,.2f}"
                    )
                with col3:
                    st.metric("Number of Predictions", len(forecast_df))

                # Visualiser
                st.markdown("### 📊 Forecast Visualization")

                fig, ax = plt.subplots(figsize=(12, 6))

                for city in selected_cities:
                    city_data = forecast_df[forecast_df["City"] == city]
                    ax.plot(
                        city_data["Date"],
                        city_data["Predicted_Sales"],
                        marker="o",
                        label=city,
                        linewidth=2,
                    )

                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Predicted Sales ($)", fontsize=12)
                ax.set_title("Sales Forecast by City", fontsize=14, fontweight="bold")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                st.pyplot(fig)

                # Afficher le tableau des données
                st.markdown("### 📋 Forecast Data")
                st.dataframe(forecast_df.sort_values(["City", "Date"]), use_container_width=True)

                # Bouton de téléchargement
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{start_date}_{n_days}days.csv",
                    mime="text/csv",
                )
            else:
                st.error(
                    "No predictions could be generated. Check if historical "
                    "data exists for selected cities."
                )


if __name__ == "__main__":
    main()
