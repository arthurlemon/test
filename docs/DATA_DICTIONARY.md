# Dataset: Données de Ventes de Magasins

## 📊 Description

Ce dataset contient les données historiques de ventes de magasins sur une période de 4 ans (2014-2017). Il est utilisé pour entraîner des modèles de Machine Learning visant à prédire les ventes futures.

## 📁 Fichier

- **Nom**: `stores_sales_forecasting.csv`
- **Taille**: ~495 KB
- **Format**: CSV (virgule comme séparateur)
- **Encodage**: UTF-8
- **Nombre de lignes**: 2,121 commandes
- **Nombre de colonnes**: 21 features potentiels
- **Source**: Inconnue, mais sûrement une version filtrée du [superstore dataset](https://www.kaggle.com/datasets/apoorvaappz/global-super-store-dataset)

## 📋 Dictionnaire des données

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| **Row ID** | Integer | Identifiant unique de la ligne | 1, 2, 3... |
| **Order ID** | String | Identifiant unique de la commande | CA-2016-152156 |
| **Order Date** | Date | Date de la commande | 11/8/2016 |
| **Ship Date** | Date | Date d'expédition | 11/11/2016 |
| **Ship Mode** | Categorical | Mode d'expédition | Second Class, First Class, Standard Class, Same Day |
| **Customer ID** | String | Identifiant unique du client | CG-12520 |
| **Customer Name** | String | Nom du client | Claire Gute |
| **Segment** | Categorical | Segment de clientèle | Consumer, Corporate, Home Office |
| **Country** | String | Pays (toujours United States) | United States |
| **City** | String | Ville | Henderson, Los Angeles... |
| **State** | String | État américain | Kentucky, California... |
| **Postal Code** | Integer | Code postal | 42420, 90032... |
| **Region** | Categorical | Région géographique | South, West, East, Central |
| **Product ID** | String | Identifiant unique du produit | FUR-BO-10001798 |
| **Category** | Categorical | Catégorie de produit | Furniture,  |
| **Sub-Category** | Categorical | Sous-catégorie | Bookcases, Chairs, Tables, Phones... |
| **Product Name** | String | Nom complet du produit | Bush Somerset Collection Bookcase |
| **Sales** 🎯 | Float | **Ventes (variable cible)** | 261.96 |
| **Quantity** | Integer | Quantité vendue | 2, 3, 5... |
| **Discount** | Float | Remise appliquée (0.0 à 1.0) | 0.0, 0.2, 0.45... |
| **Profit** | Float | Profit généré | 41.9136 (peut être négatif) |

## 🎯 Variable cible

**Sales** est la variable à prédire. Elle représente le montant total des ventes pour chaque transaction en dollars USD.

- **Minimum**: $0.44
- **Maximum**: $22,638.48
- **Moyenne**: ~$230
- **Médiane**: ~$55
