import os
import matplotlib.pyplot as plt
import random
from PIL import Image

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