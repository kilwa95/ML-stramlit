import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier



def render_knn():
    # Titre de la page
    st.title("Iris Dataset - Classification avec KNN")

    # Description du dataset
    st.write("""
    Le dataset Iris est un dataset classique en machine learning.
    Il contient des mesures de caractéristiques de fleurs d'iris de trois espèces différentes : Iris-setosa, Iris-versicolor et Iris-virginica.
    """)

    # Chargement et affichage du dataset Iris
    df = sns.load_dataset("iris")
    st.write(df)

    # Création d'un graphique de dispersion pour visualiser la relation entre la longueur des pétales et des sépales
    plt.figure(figsize = (10, 8))
    sns.scatterplot(data = df, x = df["petal_length"], y = df["sepal_length"], hue = "species")
    st.pyplot(plt)

    # Préparation des données pour le modèle KNN
    X = np.vstack((df["petal_length"], df["sepal_length"])).T
    y = np.array(df["species"]).reshape(len(df["species"]),1)

    # Division des données en ensembles d'entraînement et de test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # Création et entraînement du modèle KNN
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Affichage des résultats
    st.write(y_pred)

    # Visualisation de la matrice de confusion
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot = True, cmap = "inferno")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    st.pyplot(plt)


