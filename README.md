L’objectif est de prédire la variable Delivery_Time_min à partir de plusieurs variables explicatives (telles que la distance, le niveau de trafic, la météo, etc.) à l’aide de modèles de régression supervisée.
Deux modèles ont été comparés :

RandomForestRegressor

Support Vector Regressor (SVR)

Le but est de :

Définir un grid d’hyperparamètres pour chaque modèle.

Utiliser GridSearchCV avec validation croisée (5 folds).

Évaluer les modèles selon les métriques MAE (Mean Absolute Error) et R² (Coefficient de Détermination).

Vérifier automatiquement que la MAE maximale ne dépasse pas un seuil défini.

Justifier le choix du modèle final en fonction des performances obtenues.

1. Préparation des données

Les données ont été divisées en deux ensembles :

Train set (80%) pour l’entraînement et la recherche d’hyperparamètres.

Test set (20%) pour l’évaluation finale.

Les variables numériques ont été normalisées, et les variables catégorielles encodées lorsque nécessaire.
