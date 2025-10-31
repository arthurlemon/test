# IA x Ventes

Bonne lecture, merci pour votre temps !

1. Consulter [REPONSES.md](docs/REPONSES.md) pour les réponses aux questions.
2. Pour reproduire les résultats, voir la section [Reproduire le code](#reproduire-le-code).

## 📁 Organisation

- [docs/](docs/): réponses et data dictionary.
- [artifacts/](artifacts/): datasets (raw, processed, split), modèles (models, metrics, feature importance), artifacts d'évaluation (plot, erreurs, métriques)
- [notebooks/](notebooks/): notebook pour l'analyse exploratoire des données
- [src/](src/): modules python pour le traitement de données, l'entraînement et l'évaluation des modèles.
- [tests/](tests/): ébauches de tests

---

## 🤖 Reproduire le code

1. Cloner le repo:

```bash
git clone git@github.com:arthurlemon/test.git

cd test
```

2. Configurer l'environnement:

```bash
make setup
```

3. Reproduire la pipeline end-to-end

```bash
make pipeline
```

4. Utiliser le simulateur de prédictions (bonus)

```bash
# api locale pour utiliser le modèle
make api

# streamlit app
make dashboard
```
