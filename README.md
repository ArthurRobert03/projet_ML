## Apprentissage Profond Génératif : Inversion



![Representative image](https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/master/representative_image_512x256.png)<br>
**Picture:** Mettre la meilleure inversion

## Description
**PGGAN :**<br>
PGGAN est un modèle génératif adversarial qui apprend à produire des images de haute qualité en augmentant progressivement leur résolution. L’entraînement débute sur des images de très basse résolution, ce qui permet au générateur et au discriminateur d’apprendre les structures globales de l’image. De nouvelles couches sont ensuite ajoutées au réseau pour affiner progressivement les détails à des résolutions plus élevées. Cette stratégie rend l’apprentissage plus stable et améliore significativement la qualité et le réalisme des images générées.

**Inversion :**<br>

L’inversion d’un modèle génératif consiste à retrouver, à partir d’une image donnée, le vecteur latent qui a permis sa génération. Dans le cas de PGGAN, cette inversion repose sur l’optimisation itérative d’un vecteur latent afin de minimiser une fonction de coût mesurant l’écart entre l’image cible et l’image générée. Le générateur pré-entraîné est conservé fixe, tandis que le vecteur latent est ajusté par descente de gradient. Grâce à la structure progressive de PGGAN, l’optimisation peut être menée de manière stable, en exploitant les représentations multi-échelles du modèle pour capturer à la fois les structures globales et les détails fins de l’image.

## Structure

projet_ML/
├── config/
│ ├── train_config.yaml # Paramètres d'entraînement
│ ├── model_config.yaml # Paramètres du modèle
│ └── data_config.yaml # Paramètres des données
├── core/
│ ├── train.py # Script principal d'entraînement
│ ├── evaluate.py # Évaluation des modèles
│ ├── data_preparation.py # Préparation et nettoyage des données
│ └── inversion.py # Inversion de vecteur latent
├── data/
│ ├── raw/ # Données brutes
│ └── processed/ # Données prêtes à l'entraînement
├── gui/
│ └── visualize.py # Visualisation des résultats
├── my_models/
│ ├── pgan_generator.py # Générateur PGAN
│ └── pgan_discriminator.py # Discriminateur PGAN
├── utils/
│ ├── metrics.py # Fonctions métriques
│ └── preprocessing.py # Fonctions de preprocessing
└── README.md