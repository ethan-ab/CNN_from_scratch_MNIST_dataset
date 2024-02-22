# Projet de Reconnaissance de Chiffres Manuscrits avec CNN

## Introduction
Ce projet vise à mettre en œuvre un réseau de neurones convolutif (CNN) pour reconnaître des chiffres manuscrits à partir de l'ensemble de données MNIST.

## Description
- **Langage :** Python avec l'utilisation de bibliothèques telles que PyTorch.
- **Objectif :** Créer un modèle de CNN qui peut reconnaître les chiffres manuscrits avec une haute précision.
- **Données :** L'ensemble de données MNIST contient 60 000 images d'entraînement et 10 000 images de test de chiffres manuscrits de 0 à 9, chacun étant une image en niveaux de gris de 28x28 pixels.
- **Prétraitement des données :** Normalisation des images et création de charges de données pour l'entraînement, la validation et les tests.
- **Modèle :** Un réseau de neurones convolutif avec plusieurs couches de convolution, de max pooling et de couches entièrement connectées pour la classification.
- **Entraînement :** Optimisation du modèle à l'aide d'un algorithme d'optimisation (par exemple, Adam) et d'une fonction de perte (par exemple, Cross Entropy Loss).
- **Évaluation :** Évaluation des performances du modèle sur un ensemble de données de test distinct pour mesurer sa précision.
- **Optimisation :** Exploration des techniques d'optimisation pour améliorer les performances du modèle.

## Instructions d'utilisation
1. Télécharger ou cloner le projet depuis le référentiel.
2. Installer les dépendances Python à l'aide de `pip install -r requirements.txt`.
3. Exécuter le script `train.py` pour entraîner le modèle.

**Remarque :** Assurez-vous d'avoir une configuration Python fonctionnelle avec les bibliothèques nécessaires telles que PyTorch installées avant d'exécuter le projet.

## Auteurs
[Ethan Abimelech
]

