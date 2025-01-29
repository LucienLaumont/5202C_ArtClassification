import os
import itertools
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from PIL import Image

import gc
import tensorflow.keras.backend as K


NUM_CLASSES = 13
INPUT_SHAPE = (224,224)
INPUT_SHAPE_CNN = (128,128)
DATASET_PATH = 'dataset_600'

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

def plot_learning_curve(history):
    """
    Affiche la courbe d'apprentissage du modèle.

    Args:
        history: Objet `History` retourné par `model.fit()`.

    Returns:
        None (affiche les graphiques).
    """

    epochs = range(1, len(history.history["loss"]) + 1)

    # 🔹 Plot de la Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Évolution de la Loss")
    plt.legend()

    # 🔹 Plot de l'Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Évolution de l'Accuracy")
    plt.legend()

    plt.show()


def plot_class_performance(model, dataset, class_names):
    """
    Affiche la précision, le recall et le F1-score pour chaque classe.

    Args:
        model: Modèle entraîné.
        dataset: Dataset de test (ou validation) contenant les features et labels.
        class_names (list): Liste des noms des classes.

    Returns:
        None (affiche le graphique).
    """

    # Récupérer les vraies étiquettes et les prédictions
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_pred_probs = model.predict(dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calcul du rapport de classification
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Extraction des métriques par classe
    precision = [report[cls]["precision"] for cls in class_names]
    recall = [report[cls]["recall"] for cls in class_names]
    f1_score = [report[cls]["f1-score"] for cls in class_names]

    # Tracé du graphique
    x = np.arange(len(class_names))
    width = 0.25  # Largeur des barres

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label="Précision", alpha=0.8)
    plt.bar(x, recall, width, label="Recall", alpha=0.8)
    plt.bar(x + width, f1_score, width, label="F1-score", alpha=0.8)

    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Performance du modèle par classe")
    plt.xticks(ticks=x, labels=class_names, rotation=45)
    plt.legend()
    plt.show()


def sample_images_class_count():
     
     # Initialiser les données pour le graphique
    class_counts = {}
    sample_images = []
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        files = os.listdir(class_path)
        class_counts[class_name] = len(files)
    
    # Prendre quelques images aléatoires pour l'affichage
    sample_images.extend([os.path.join(class_path, f) for f in random.sample(files, min(3, len(files)))])

    return sample_images,class_counts

def ShowSamples(T):
  plt.figure(figsize=(10, 10))
  plt.subplots_adjust(top=1)
  for images, labels in T.take(1):
          for i in range(len(images)):
                  ax = plt.subplot(10, 10, i + 1)
                  plt.imshow(np.array(images[i]).astype("uint8"))
                  plt.title(int(labels[i]))
                  plt.axis("off")


def dataset_augmentation(data_augmentation_layers,images):
 
    for layer in data_augmentation_layers:
                images = layer(images)
    return images

import itertools
import pandas as pd

def generate_hyperparameter_combinations(learning_rates,epochs):
    """
    Génère des combinaisons d'hyperparamètres pour tester différents modèles.

    Returns:
        list[dict]: Liste de dictionnaires contenant les combinaisons d'hyperparamètres.
    """
    dropout_options = [True, False]
    data_augmentation = [True, False]
    combinations = list(itertools.product(epochs, learning_rates, dropout_options, data_augmentation))
    return [
        {"epochs" : ep,"learning_rate": lr, "dropout": do, "data_augmentation": da}
        for ep, lr, do, da in combinations
    ]

def build_fc_model(dropout = False):
    """
    Construit un modèle entièrement connecté (Fully Connected Network) avec des améliorations pour mieux capturer les caractéristiques des données et limiter le surapprentissage.

    Parameters:
        input_shape (tuple): La forme des données d'entrée sans les canaux (par ex., (128, 128) pour des images RGB de 128x128).
        num_classes (int): Nombre de classes à prédire (13 dans ce cas).

    Returns:
        model: Modèle Keras compilé.
    """
    # Compléter dynamiquement la troisième dimension (canaux de couleur)
    input_shape = (*INPUT_SHAPE, 3)  # Ajoute la troisième dimension

    # Construction du modèle
    model = Sequential()

    # Entrée et prétraitement
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))  # Normalisation des pixels entre 0 et 1

    # Extraction de caractéristiques via plusieurs couches denses
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    if dropout:    
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.2))

    return model

def build_cnn_model(dropout = False):
    """
    Construit un modèle CNN avec des améliorations pour mieux capturer les caractéristiques des données et limiter le surapprentissage.

    Parameters:
        dropout (Bool): Activation du dropout ou non.

    Returns:
        model: Modèle Keras compilé.
    """
    # Compléter dynamiquement la troisième dimension (canaux de couleur)
    input_shape = (*INPUT_SHAPE, 3)  # Ajoute la troisième dimension

    # Construction du modèle
    model = Sequential()

    # Entrée et prétraitement
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1.0 / 255))  # Normalisation des pixels entre 0 et 1

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization()),
    if dropout:
        model.add(layers.Dropout(0.3)) 

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization()) 
    if dropout:    
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization()),
    if dropout:
        model.add(layers.Dropout(0.2))

    return model

def build_mobilenetv2(dropout=False, dropout_rate=0.3):
    """
    Utilise MobileNetV2 préentraîné pour réaliser un transfert learning.

    Args:
        dropout (bool): Active ou désactive le dropout.
        dropout_rate (float): Taux de dropout utilisé si dropout=True.

    Returns:
        model: Modèle Keras compilé.
    """
    
    # Charger MobileNetV2 avec des poids pré-entraînés
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE + (3,), 
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential()

    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    if dropout:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(128))
    if dropout:
        model.add(layers.Dropout(dropout_rate))

    return model



def train_model(model_type, train, valid, params):
    """
    Entraîne un modèle et retourne le modèle, les métriques et l'historique d'entraînement.

    Args:
        model_type (str): Type de modèle à entraîner ("FC", "CNN", "TL").
        dataset_dictionnary (dict): Dictionnaire contenant les datasets train et valid.
        params (dict): Dictionnaire des hyperparamètres.

    Returns:
        model: Le modèle entraîné.
        dict: Contient les hyperparamètres et les métriques finales.
        history: L'objet `History` retourné par `model.fit()`.
    """

    model_builders = {
        "FC": build_fc_model,
        "CNN": build_cnn_model,
        "TL": build_mobilenetv2
    }

    if model_type not in model_builders:
        raise ValueError(f"Type de modèle inconnu : {model_type}. Choisissez parmi {list(model_builders.keys())}.")

    print(f"🚀 Initialisation du modèle {model_type}...")

    model = model_builders[model_type](params["dropout"])

    if model_type == "TL":
        print("🔄 Prétraitement des données pour Transfer Learning...")
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        train = train.map(lambda x, y: (preprocess_input(x), y))
        valid = valid.map(lambda x, y: (preprocess_input(x), y))
        
    # Couche de sortie
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))  # Activation pour classification multi-classe
    
    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),  # Taux d'apprentissage modifiable
        loss=SparseCategoricalCrossentropy(from_logits=False),  # from_logits=False pour softmax
        metrics=['accuracy']
    )
 

    print("📊 Début de l'entraînement...")

    history = model.fit(
        train,
        validation_data=valid,
        epochs=params['epochs'],
        verbose=1
    )

    print("✅ Entraînement terminé.")

    metrics = {
        **params,
        "final_loss": history.history["loss"][-1],
        "final_accuracy": history.history["accuracy"][-1],
        "final_val_loss": history.history["val_loss"][-1],
        "final_val_accuracy": history.history["val_accuracy"][-1]
    }


    return model, metrics, history


def train_all_models(model_type, train,valid,augmented_train, hyperparams):
    """
    Itère sur les combinaisons d'hyperparamètres, entraîne les modèles,
    et retourne un DataFrame avec les performances ainsi que le meilleur modèle et son historique.

    Args:
        model_type (str): Type de modèle ("FC", "CNN", "TL").
        dataset_dictionnary (dict[Dataset]): Dictionnaire des datasets train, valid et augmentation.
        hyperparams (list[dict]): Liste des combinaisons d'hyperparamètres.

    Returns:
        pd.DataFrame: DataFrame contenant les performances pour chaque configuration.
        model: Le modèle ayant la meilleure `final_val_accuracy`.
        history: L'historique d'entraînement du meilleur modèle.
    """

    results = []
    best_model = None
    best_history = None
    best_accuracy = -1  # Initialisation pour comparaison

    for params in hyperparams:
        print(f"🔍 Entraînement avec les paramètres : {params}")

        if params["data_augmentation"]:
            model, metrics, history = train_model(model_type, augmented_train,valid, params)
        else:
            model, metrics, history = train_model(model_type, train,valid, params)

        results.append(metrics)

        if metrics["final_val_accuracy"] > best_accuracy:
            best_accuracy = metrics["final_val_accuracy"]
            best_model = model
            best_history = history  # Sauvegarde de l'historique du meilleur modèle

    df_results = pd.DataFrame(results)

    print(f"🏆 Meilleur modèle : {best_accuracy*100:.2f}% de final_val_accuracy")

    return df_results, best_model, best_history
