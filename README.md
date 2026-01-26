## Apprentissage Profond Génératif : Inversion



![Representative image](data/presentation.png)<br>
**Picture:** Résultats de l'inversion (Image de base à gauche et Inversion à droite)

## Description
**PGGAN :**<br>

PGGAN est un modèle génératif adversarial qui apprend à produire des images de haute qualité en augmentant progressivement leur résolution. L’entraînement débute sur des images de très basse résolution, ce qui permet au générateur et au discriminateur d’apprendre les structures globales de l’image. De nouvelles couches sont ensuite ajoutées au réseau pour affiner progressivement les détails à des résolutions plus élevées. Cette stratégie rend l’apprentissage plus stable et améliore significativement la qualité et le réalisme des images générées.

**Inversion :**<br>

L’inversion d’un modèle génératif consiste à retrouver, à partir d’une image donnée, le vecteur latent qui a permis sa génération. Dans le cas de PGGAN, cette inversion repose sur l’optimisation itérative d’un vecteur latent afin de minimiser une fonction de coût mesurant l’écart entre l’image cible et l’image générée. Le générateur pré-entraîné est conservé fixe, tandis que le vecteur latent est ajusté par descente de gradient. Grâce à la structure progressive de PGGAN, l’optimisation peut être menée de manière stable, en exploitant les représentations multi-échelles du modèle pour capturer à la fois les structures globales et les détails fins de l’image.

## Structure

| Fichier | Description |Fonctions |
|----|-------------|----------------------|
| `core/train.py` | Script principal pour l’entraînement des modèles PGGAN | Trainer, train, save_checkpoint |
| `core/evaluate.py` | Évaluation des modèles sur les jeux de test | Evaluator, compute_metrics, generate_samples |
| `core/data_preparation.py` | Prépare et nettoie les données pour l’entraînement | load_data, normalize_data, split_dataset | 
| `core/inversion.py`| Inversion d’un vecteur latent à partir d’une image cible	| invert_image, latent_optimization, compute_loss | 
| `my_models/pgan_generator.py` | Définition du générateur PGGAN | PGANGenerator, forward | 
| `my_models/pgan_discriminator.py` | Définition du discriminateur PGGAN | PGANDiscriminator, forward |
| `utils/preprocessing.py` | Fonctions utilitaires pour le prétraitement des données | resize_images, normalize, augment_data | 
| `utils/metrics.py` | Calcul des métriques et pertes pour l’entraînement et l’évaluation | compute_mse, compute_lpips, compute_fid | 
| `gui/visualize.py` | Visualisation des images reconstruites | gui.app |
