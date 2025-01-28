import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import numpy as np

# Afficher quelques images de la base
def show_sample_images(images, title="Exemples d'images"):
    plt.figure(figsize=(12, 8))
    for i, img_path in enumerate(images[:9]):
        plt.subplot(3, 3, i + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(os.path.basename(os.path.dirname(img_path)))
        plt.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.show()

# Générer un graphique en camembert
def plot_pie_chart(data, title="Distribution des images par classes"):
    labels = list(data.keys())
    sizes = list(data.values())
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.show()

def ShowSamples(T):
  plt.figure(figsize=(10, 10))
  plt.subplots_adjust(top=1)
  for images, labels in T.take(1):
          for i in range(len(images)):
                  ax = plt.subplot(10, 10, i + 1)
                  plt.imshow(np.array(images[i]).astype("uint8"))
                  plt.title(int(labels[i]))
                  plt.axis("off")


def normalize_datasets(train_dataset, validation_dataset):
    """
    Prépare les ensembles d'entraînement et de validation en normalisant les images.

    Parameters:
        train_dataset: Ensemble d'entraînement brut.
        validation_dataset: Ensemble de validation brut.

    Returns:
        train_dataset: Ensemble d'entraînement prétraité.
        validation_dataset: Ensemble de validation prétraité.
    """
    normalization_layer = lambda x, y: (x / 255.0, y)  # Normalisation des pixels

    train_dataset = train_dataset.map(normalization_layer)
    validation_dataset = validation_dataset.map(normalization_layer)

    return train_dataset, validation_dataset