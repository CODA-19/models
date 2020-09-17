## Federated multi-input CNN training for mortality prediction in patients with COVID-19

Modèle multi-input (time series = laboratoires, signes vitaux + 2-dimensional imaging = rayon X poumon) pour prédire la mortalité à 90 jours.

### Population

- Tous les patients ayant testé positif pour le COVID qui ont eu un épisode-patient (urgence, étage ou SI) dans un délai de 21 jours du test COVID (n=343). Un seul épisode index par patient est inclus (le plus rapproché du premier test COVID positif - les autres sont éliminés). 
- Sont inclus les patients ayant eu un rayon X pulmonaire dans les 48 heures du début de l'épisode index. On exclut les patients sans imagerie.

### Preprocessing

Pour les patients qui demeurent (n = 188 avec 45 décès à 90 jours):

- On extrait le subset des patients décédés et on crée un outcome binaire "décès à 90 jours" basé sur le délai entre le décès et le début de l'épisode index.
- On récupère et préformatte les rayon X pulmonaires (crop 1/4 de tous les côtés, scale à 192x192, égalisation de l'histogramme). 
- On extrait des labos (9) et signes vitaux (7) que j'ai choisi à priori (basé sur les scores classiques de sévérité et les études publiées chez les patients COVID). Donc 16 observations au total, je considère des fenêtres de 6h sur la période de 0 à 72h (total 12 time points x 16 observations = 192 observations par patient).

Les données manquantes sont imputées avec un denoising autoencoder (single imputation).

### Model

Modèle multi-input (time series = laboratoires, signes vitaux + 2-dimensional imaging = rayon X poumon) pour prédire la mortalité à 90 jours. Il s'agit de deux 2-D CNN simples (2 convolutions chaque) implémentés avec Keras (j'ai attaché le diagramme.) Le premier CNN prend les labos et signes vitaux en multivarié (16 obs x 12 time points) et le deuxième CNN prend les rayons X poumons (192 x 192 pixels). Les deux CNN sont essentiellement identiques sauf pour les kernel sizes (4 et 2 pour les time series; 3 et 3 pour l'imagerie) et le dropout (uniform random dropout pour tabulaire et Gaussian dropout pour imagerie).

Chaque CNN émet un output de taille (32 x batch_size), et les outputs des 2 CNN sont ensuite fusionnés, avant de passer dans un dense layer pour la prédiction finale.

![alt text](model.png "Model structure")

### Evaluation

L'évaluation du modèle se fait avec un K-fold (K = 5) stratifié (% des deux classes du outcome similaire dans tous les folds). Le split des folds est fait après un shuffling aléatoire des données. À l'intérieur de chaque fold, les poids des classes sont pondérées dans le loss function par l'inverse de leur fréquence d'apparition dans le fold. Ces trois trucs aident à pallier au class imbalance sans faire de resampling.

Pour chacun des 5 folds (80/20), la portion de 20% est conservée comme test set. La portion de 80% est séparée en 70% de training data et 30% de validation. Le modèle est donc fitté et évalué 5 fois et les résultats sont pondérés. J'ai mis du early stopping sur tous les modèles.

### Results

![alt text](results.png "Results")
