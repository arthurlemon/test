---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: sales-forecasting
    language: python
    name: python3
---

## Analyse Exploratoire du Dataset

```python
# Imports
%reload_ext autoreload
%autoreload 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ydata_profiling import ProfileReport

from src.etl import DataLoader

# Configuration des graphiques
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
%matplotlib inline

DATA_FOLDER = Path("../artifacts/data")

print("✅ Imports réussis")
```

### 1. Chargement des Données

```python
data_path = DATA_FOLDER / "raw/stores_sales_forecasting.csv"
loader = DataLoader(str(data_path))
df_sales = loader.load_data()

# Informations de base
info = loader.get_basic_info()
print("Dataset chargé:")
print(f"  - Lignes: {info['n_lignes']:,}")
print(f"  - Colonnes: {info['n_colonnes']}")
print(f"  - Mémoire: {info['memoire_mb']:.2f} MB")

# Aperçu
print("\n5 premières lignes:")
df_sales.head()
```

### 2. EDA

Génération d'un rapport complet et interactif pour identifier automatiquement:
- Statistiques descriptives
- Distributions
- Corrélations
- Valeurs manquantes
- Outliers
- Alertes de qualité

```python
# Générer le rapport ydata-profiling
profile = ProfileReport(
    df_sales,
    title="Analyse Exploratoire - Données de Ventes",
    explorative=True,
)

report_path = DATA_FOLDER / "raw/eda_report.html"
profile.to_file(report_path)

print(f"\n✅ Rapport sauvegardé: {report_path}. Peut être ouvert dans un navigateur.")
```

```python
# run pour ouvrir le rapport ici, sinon ouvrir le fichier HTML dans un navigateur
profile.to_notebook_iframe()
```

```python
# statistics
df_sales.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
```

```python
df_sales.isnull().sum()
```

```python
### 3. Analyse des Outliers avec Boxplots

continuous_vars = df_sales.select_dtypes(include=["float64", "int64"]).columns.tolist()
continuous_vars = [col for col in continuous_vars if col not in ["Row ID", "Postal Code"]]

categorical_vars = ["Segment", "Ship Mode", "State", "Region", "Category", "Sub-Category"]

print(f"Variables continues à analyser: {continuous_vars}")
print(f"Variables catégorielles pour groupement: {categorical_vars[:4]}")

# Boxplots pour les variables continues
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Boxplots des Variables Continues - Détection des Outliers", fontsize=16, fontweight="bold"
)

for idx, var in enumerate(continuous_vars[:4]):  # Sales, Quantity, Discount, Profit
    ax = axes[idx // 2, idx % 2]
    bp = ax.boxplot(df_sales[var].dropna(), vert=True, patch_artist=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.set_ylabel("Valeur", fontsize=10)
    ax.set_title(f"{var}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    Q1 = df_sales[var].quantile(0.25)
    Q3 = df_sales[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_sales[(df_sales[var] < lower_bound) | (df_sales[var] > upper_bound)][var]

    ax.text(
        0.98,
        0.98,
        f"Outliers: {len(outliers)} ({len(outliers) / len(df_sales) * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Boxplots de Sales par Variables Catégorielles", fontsize=16, fontweight="bold")

for idx, cat_var in enumerate(categorical_vars[:4]):
    ax = axes[idx // 2, idx % 2]

    data_to_plot = [
        df_sales[df_sales[cat_var] == cat]["Sales"].values for cat in df_sales[cat_var].unique()
    ]
    labels = df_sales[cat_var].unique()

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Sales", fontsize=11)
    ax.set_xlabel(cat_var, fontsize=11)
    ax.set_title(f"Distribution de Sales par {cat_var}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("RÉSUMÉ DES OUTLIERS (méthode IQR: Q1-1.5*IQR et Q3+1.5*IQR)")
print("=" * 80)

outlier_summary = []
for var in continuous_vars[:4]:
    Q1 = df_sales[var].quantile(0.25)
    Q3 = df_sales[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_low = df_sales[df_sales[var] < lower_bound]
    outliers_high = df_sales[df_sales[var] > upper_bound]
    total_outliers = len(outliers_low) + len(outliers_high)

    outlier_summary.append(
        {
            "Variable": var,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers Bas": len(outliers_low),
            "Outliers Haut": len(outliers_high),
            "Total Outliers": total_outliers,
            "Pourcentage": f"{total_outliers / len(df_sales) * 100:.2f}%",
        }
    )

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
print("=" * 80)
```

```python
### 4. Distribution des Ventes par Variables Catégorielles

categorical_fields = ["Segment", "Ship Mode", "Region", "State", "City", "Sub-Category"]

print("=" * 100)
print("DISTRIBUTION DES VENTES PAR VARIABLE CATÉGORIELLE")
print("=" * 100)

for field in categorical_fields:
    print(f"\n{'=' * 100}")
    print(f"📊 {field.upper()}")
    print(f"{'=' * 100}")

    counts = df_sales[field].value_counts()
    percentages = (counts / len(df_sales) * 100).round(2)

    summary_df = pd.DataFrame(
        {
            "Catégorie": counts.index,
            "Nombre de Ventes": counts.values,
            "Pourcentage (%)": percentages.values,
        }
    )

    total_row = pd.DataFrame(
        {"Catégorie": ["TOTAL"], "Nombre de Ventes": [counts.sum()], "Pourcentage (%)": [100.00]}
    )
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)

    print(summary_df.to_string(index=False))
    print(f"\nNombre de catégories uniques: {len(counts)}")

print("\n" + "=" * 100)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Distribution des Ventes par Variables Catégorielles", fontsize=16, fontweight="bold")

for idx, field in enumerate(categorical_fields):
    ax = axes[idx // 3, idx % 3]

    # Pour City, limiter aux 10 villes les plus fréquentes pour la lisibilité
    if field == "City":
        counts = df_sales[field].value_counts().head(10)
        title_suffix = " (Top 10)"
    else:
        counts = df_sales[field].value_counts()
        title_suffix = ""

    bars = ax.bar(
        range(len(counts)), counts.values, color=sns.color_palette("viridis", len(counts))
    )

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Nombre de Ventes", fontsize=10)
    ax.set_title(f"{field}{title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for i, (bar, value) in enumerate(zip(bars, counts.values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(value)}\n({value / len(df_sales) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
plt.show()

print("\n" + "=" * 100)
print("RÉSUMÉ GLOBAL - CARDINALITÉ DES VARIABLES CATÉGORIELLES")
print("=" * 100)

cardinality_summary = []
for field in categorical_fields:
    unique_count = df_sales[field].nunique()
    most_common = df_sales[field].value_counts().index[0]
    most_common_count = df_sales[field].value_counts().values[0]
    most_common_pct = most_common_count / len(df_sales) * 100

    cardinality_summary.append(
        {
            "Variable": field,
            "Valeurs Uniques": unique_count,
            "Catégorie la Plus Fréquente": most_common,
            "Nombre": most_common_count,
            "Pourcentage (%)": f"{most_common_pct:.2f}%",
        }
    )

cardinality_df = pd.DataFrame(cardinality_summary)
print(cardinality_df.to_string(index=False))
print("=" * 100)
```

```python
### 5. Analyse des Séries Temporelles

# Convertir les dates en datetime
df_sales["Order Date"] = pd.to_datetime(df_sales["Order Date"])
df_sales["Ship Date"] = pd.to_datetime(df_sales["Ship Date"])

# Trier par date pour l'analyse temporelle
df_sorted = df_sales.sort_values("Order Date")

print("=" * 100)
print("ANALYSE TEMPORELLE DES VENTES")
print("=" * 100)
print(
    f"\nPériode des données: {df_sorted['Order Date'].min().date()} à {df_sorted['Order Date'].max().date()}"
)
print(f"Durée totale: {(df_sorted['Order Date'].max() - df_sorted['Order Date'].min()).days} jours")

# Agrégations temporelles
df_sorted["Year"] = df_sorted["Order Date"].dt.year
df_sorted["Quarter"] = df_sorted["Order Date"].dt.quarter
df_sorted["Month"] = df_sorted["Order Date"].dt.to_period("M")
df_sorted["YearMonth"] = df_sorted["Order Date"].dt.to_period("M")

# Créer les visualisations
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Analyse des Séries Temporelles - Sales", fontsize=16, fontweight="bold")

# 1. Ventes mensuelles
ax1 = axes[0, 0]
monthly_sales = df_sorted.groupby("YearMonth")["Sales"].agg(["sum", "count", "mean"]).reset_index()
monthly_sales["YearMonth"] = monthly_sales["YearMonth"].astype(str)

ax1.plot(range(len(monthly_sales)), monthly_sales["sum"], marker="o", linewidth=2, markersize=4)
ax1.set_xlabel("Mois", fontsize=10)
ax1.set_ylabel("Ventes Totales ($)", fontsize=10)
ax1.set_title("Évolution des Ventes Mensuelles", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis="x", rotation=45, labelsize=7)
# Afficher seulement quelques labels pour la lisibilité
step = max(1, len(monthly_sales) // 10)
ax1.set_xticks(range(0, len(monthly_sales), step))
ax1.set_xticklabels(monthly_sales["YearMonth"][::step], rotation=45, ha="right")

# Ajouter rolling average
if len(monthly_sales) >= 3:
    rolling_avg = monthly_sales["sum"].rolling(window=3, center=True).mean()
    ax1.plot(
        range(len(monthly_sales)),
        rolling_avg,
        "r--",
        linewidth=2,
        label="Moyenne mobile (3 mois)",
        alpha=0.7,
    )
    ax1.legend()

# 2. Ventes par trimestre
ax2 = axes[0, 1]
quarterly_sales = df_sorted.groupby(["Year", "Quarter"])["Sales"].sum().reset_index()
quarterly_sales["Quarter_Label"] = (
    quarterly_sales["Year"].astype(str) + "-Q" + quarterly_sales["Quarter"].astype(str)
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
bars = ax2.bar(
    range(len(quarterly_sales)),
    quarterly_sales["Sales"],
    color=[colors[q - 1] for q in quarterly_sales["Quarter"]],
)
ax2.set_xlabel("Trimestre", fontsize=10)
ax2.set_ylabel("Ventes Totales ($)", fontsize=10)
ax2.set_title("Ventes par Trimestre", fontsize=12, fontweight="bold")
ax2.set_xticks(range(len(quarterly_sales)))
ax2.set_xticklabels(quarterly_sales["Quarter_Label"], rotation=45, ha="right", fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# Ajouter les valeurs sur les barres
for i, (bar, value) in enumerate(zip(bars, quarterly_sales["Sales"])):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"${value:,.0f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# 3. Comparaison année par année
ax3 = axes[1, 0]
yearly_sales = df_sorted.groupby("Year")["Sales"].agg(["sum", "count", "mean"]).reset_index()

ax3_twin = ax3.twinx()
bars = ax3.bar(
    yearly_sales["Year"], yearly_sales["sum"], alpha=0.7, color="steelblue", label="Ventes totales"
)
line = ax3_twin.plot(
    yearly_sales["Year"],
    yearly_sales["count"],
    "ro-",
    linewidth=2,
    markersize=8,
    label="Nombre de transactions",
)

ax3.set_xlabel("Année", fontsize=10)
ax3.set_ylabel("Ventes Totales ($)", fontsize=10, color="steelblue")
ax3_twin.set_ylabel("Nombre de Transactions", fontsize=10, color="red")
ax3.set_title("Évolution Annuelle des Ventes", fontsize=12, fontweight="bold")
ax3.tick_params(axis="y", labelcolor="steelblue")
ax3_twin.tick_params(axis="y", labelcolor="red")
ax3.grid(True, alpha=0.3, axis="y")

# Légende combinée
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# 4. Distribution des ventes par trimestre (saisonnalité)
ax4 = axes[1, 1]
quarter_dist = df_sorted.groupby("Quarter")["Sales"].agg(["sum", "mean", "count"]).reset_index()

bars = ax4.bar(
    quarter_dist["Quarter"], quarter_dist["sum"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)
ax4.set_xlabel("Trimestre", fontsize=10)
ax4.set_ylabel("Ventes Totales ($)", fontsize=10)
ax4.set_title("Saisonnalité - Ventes par Trimestre (Toutes Années)", fontsize=12, fontweight="bold")
ax4.set_xticks(quarter_dist["Quarter"])
ax4.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
ax4.grid(True, alpha=0.3, axis="y")

# Ajouter les valeurs et pourcentages
total_sales = quarter_dist["sum"].sum()
for bar, value in zip(bars, quarter_dist["sum"]):
    height = bar.get_height()
    pct = value / total_sales * 100
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"${value:,.0f}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

# Statistiques récapitulatives
print("\n" + "=" * 100)
print("STATISTIQUES TEMPORELLES")
print("=" * 100)

print("\n📊 Ventes par Année:")
yearly_summary = df_sorted.groupby("Year")["Sales"].agg(["sum", "count", "mean"]).round(2)
yearly_summary.columns = ["Total ($)", "Nombre de Transactions", "Vente Moyenne ($)"]
print(yearly_summary.to_string())

print("\n📊 Ventes par Trimestre (Agrégé toutes années):")
quarter_summary = df_sorted.groupby("Quarter")["Sales"].agg(["sum", "count", "mean"]).round(2)
quarter_summary.columns = ["Total ($)", "Nombre de Transactions", "Vente Moyenne ($)"]
quarter_summary.index = ["Q1", "Q2", "Q3", "Q4"]
quarter_summary["Pourcentage (%)"] = (
    quarter_summary["Total ($)"] / quarter_summary["Total ($)"].sum() * 100
).round(2)
print(quarter_summary.to_string())

# Insights clés
print("\n" + "=" * 100)
print("🔍 INSIGHTS CLÉS")
print("=" * 100)

best_quarter = quarter_summary["Total ($)"].idxmax()
worst_quarter = quarter_summary["Total ($)"].idxmin()
q_diff = (
    quarter_summary.loc[best_quarter, "Total ($)"] / quarter_summary.loc[worst_quarter, "Total ($)"]
    - 1
) * 100

print(
    f"• Trimestre le plus performant: {best_quarter} (${quarter_summary.loc[best_quarter, 'Total ($)']:,.0f})"
)
print(
    f"• Trimestre le moins performant: {worst_quarter} (${quarter_summary.loc[worst_quarter, 'Total ($)']:,.0f})"
)
print(f"• Écart de performance: {q_diff:.1f}% entre le meilleur et le pire trimestre")

if len(yearly_sales) > 1:
    yoy_growth = (
        (yearly_sales["sum"].iloc[-1] / yearly_sales["sum"].iloc[0])
        ** (1 / (len(yearly_sales) - 1))
        - 1
    ) * 100
    print(f"• Croissance annuelle moyenne: {yoy_growth:.1f}%")

print("=" * 100)
```

```python
### 6. Décomposition de la Saisonnalité des Ventes

from statsmodels.tsa.seasonal import seasonal_decompose

# Créer une série temporelle agrégée par date (moyenne quotidienne)
# Pour avoir une série régulière, nous allons agréger par jour
daily_sales = df_sorted.groupby(df_sorted["Order Date"].dt.date)["Sales"].sum().reset_index()
daily_sales.columns = ["Date", "Sales"]
daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
daily_sales = daily_sales.set_index("Date")
daily_sales = daily_sales.sort_index()

# Remplir les dates manquantes avec interpolation pour avoir une série continue
daily_sales = daily_sales.asfreq("D")
daily_sales["Sales"] = daily_sales["Sales"].fillna(0)

print("=" * 100)
print("DÉCOMPOSITION DE LA SAISONNALITÉ DES VENTES")
print("=" * 100)
print(f"\nPériode d'analyse: {daily_sales.index.min().date()} à {daily_sales.index.max().date()}")
print(f"Nombre de jours: {len(daily_sales)}")
print(f"Ventes quotidiennes moyennes: ${daily_sales['Sales'].mean():,.2f}")
print(f"Écart-type des ventes quotidiennes: ${daily_sales['Sales'].std():,.2f}")

# Décomposition saisonnière
# Utilisation d'une période de 30 jours (approximativement un mois) pour capturer la saisonnalité mensuelle
# et modèle multiplicatif pour capturer la variation proportionnelle
print("\n🔍 Décomposition en cours...")

# Appliquer un lissage avec rolling average pour réduire le bruit
daily_sales_smooth = daily_sales["Sales"].rolling(window=7, center=True).mean()
daily_sales_smooth = daily_sales_smooth.fillna(daily_sales["Sales"])

# Décomposition avec modèle multiplicatif (meilleur pour les données de ventes)
try:
    decomposition = seasonal_decompose(
        daily_sales_smooth,
        model="multiplicative",
        period=30,  # Période mensuelle
        extrapolate_trend="freq",
    )
    decomp_model = "multiplicatif"
except:
    # Si le modèle multiplicatif échoue (par exemple, avec des valeurs nulles), utiliser additif
    decomposition = seasonal_decompose(
        daily_sales["Sales"], model="additive", period=30, extrapolate_trend="freq"
    )
    decomp_model = "additif"

print(f"✅ Décomposition terminée (modèle {decomp_model})")

# Visualisation des composantes
fig, axes = plt.subplots(4, 1, figsize=(18, 14))
fig.suptitle(
    f"Décomposition Saisonnière des Ventes (Modèle {decomp_model.capitalize()})",
    fontsize=16,
    fontweight="bold",
)

# 1. Série originale
axes[0].plot(decomposition.observed, linewidth=1.5, color="#2c3e50")
axes[0].set_ylabel("Ventes Observées ($)", fontsize=11, fontweight="bold")
axes[0].set_title("Série Temporelle Originale", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis="x", rotation=45)

# Statistiques de la série originale
mean_val = decomposition.observed.mean()
axes[0].axhline(
    y=mean_val,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Moyenne: ${mean_val:,.0f}",
    alpha=0.7,
)
axes[0].legend(loc="upper right")

# 2. Tendance
axes[1].plot(decomposition.trend, linewidth=2, color="#e74c3c")
axes[1].set_ylabel("Tendance", fontsize=11, fontweight="bold")
axes[1].set_title("Tendance à Long Terme", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis="x", rotation=45)

# Calculer le taux de croissance de la tendance
trend_values = decomposition.trend.dropna()
if len(trend_values) > 1:
    trend_change = ((trend_values.iloc[-1] / trend_values.iloc[0]) - 1) * 100
    trend_direction = "↗️ Croissance" if trend_change > 0 else "↘️ Décroissance"
    axes[1].text(
        0.02,
        0.95,
        f"{trend_direction}: {abs(trend_change):.1f}%",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        verticalalignment="top",
    )

# 3. Saisonnalité
axes[2].plot(decomposition.seasonal, linewidth=1.5, color="#27ae60")
axes[2].set_ylabel("Saisonnalité", fontsize=11, fontweight="bold")
axes[2].set_title("Composante Saisonnière (Cycle de 30 jours)", fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis="x", rotation=45)

# Amplitude de la saisonnalité
seasonal_amplitude = decomposition.seasonal.max() - decomposition.seasonal.min()
axes[2].text(
    0.02,
    0.95,
    f"Amplitude: {seasonal_amplitude:.2f}",
    transform=axes[2].transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    verticalalignment="top",
)

# 4. Résidus (bruit)
axes[3].plot(decomposition.resid, linewidth=0.8, color="#95a5a6", alpha=0.7)
axes[3].set_ylabel("Résidus", fontsize=11, fontweight="bold")
axes[3].set_xlabel("Date", fontsize=11, fontweight="bold")
axes[3].set_title("Résidus (Bruit Aléatoire)", fontsize=12)
axes[3].grid(True, alpha=0.3)
axes[3].tick_params(axis="x", rotation=45)
axes[3].axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)

# Statistiques des résidus
resid_std = decomposition.resid.std()
axes[3].text(
    0.02,
    0.95,
    f"Écart-type: {resid_std:.2f}",
    transform=axes[3].transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    verticalalignment="top",
)

plt.tight_layout()
plt.show()

# Analyse statistique des composantes
print("\n" + "=" * 100)
print("📊 STATISTIQUES DES COMPOSANTES")
print("=" * 100)

components_stats = pd.DataFrame(
    {
        "Composante": ["Observée", "Tendance", "Saisonnalité", "Résidus"],
        "Moyenne": [
            decomposition.observed.mean(),
            decomposition.trend.mean(),
            decomposition.seasonal.mean(),
            decomposition.resid.mean(),
        ],
        "Écart-type": [
            decomposition.observed.std(),
            decomposition.trend.std(),
            decomposition.seasonal.std(),
            decomposition.resid.std(),
        ],
        "Min": [
            decomposition.observed.min(),
            decomposition.trend.min(),
            decomposition.seasonal.min(),
            decomposition.resid.min(),
        ],
        "Max": [
            decomposition.observed.max(),
            decomposition.trend.max(),
            decomposition.seasonal.max(),
            decomposition.resid.max(),
        ],
    }
)

print(components_stats.to_string(index=False))

# Insights clés
print("\n" + "=" * 100)
print("🔍 INSIGHTS CLÉS SUR LA DÉCOMPOSITION")
print("=" * 100)

# Force de la tendance
trend_contribution = (decomposition.trend.std() / decomposition.observed.std()) * 100
print(f"• Contribution de la tendance à la variance: {trend_contribution:.1f}%")

# Force de la saisonnalité
seasonal_contribution = (decomposition.seasonal.std() / decomposition.observed.std()) * 100
print(f"• Contribution de la saisonnalité à la variance: {seasonal_contribution:.1f}%")

# Bruit résiduel
noise_contribution = (decomposition.resid.std() / decomposition.observed.std()) * 100
print(f"• Contribution du bruit résiduel à la variance: {noise_contribution:.1f}%")

# Interprétation
print("\n💡 Interprétation:")
if trend_contribution > 50:
    print("  - La série présente une forte tendance à long terme")
elif trend_contribution > 30:
    print("  - La série présente une tendance modérée")
else:
    print("  - La tendance à long terme est faible")

if seasonal_contribution > 30:
    print("  - La saisonnalité est très marquée (variations cycliques importantes)")
elif seasonal_contribution > 15:
    print("  - La saisonnalité est modérée")
else:
    print("  - La saisonnalité est faible")

if noise_contribution > 40:
    print("  - Le bruit est élevé (série volatile avec beaucoup d'irrégularités)")
elif noise_contribution > 20:
    print("  - Le bruit est modéré")
else:
    print("  - Le bruit est faible (série relativement stable)")

print("=" * 100)
```
