import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def render_decision_tree_classifier():
    # Chargement des données
    df = pd.read_csv('data/titanic.csv')
    
    # Affichage du titre
    st.title("Classification avec Arbre de Décision - Dataset Titanic")
    
    
    # Nettoyage du jeu de données
    df.dropna(inplace = True)

    # Affichage toutes les lignes du dataset
    st.dataframe(df)

    # Sélection des caractéristiques (features)
    X = df[["Age", "Pclass"]].values

    # Sélection de la variable cible (target)
    y = df["Survived"].values

    # Division des données en ensembles d'entraînement et de test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création et entraînement du modèle de classification
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Précision du modèle: {accuracy:.2f}")

    # Création d'un scatter plot pour visualiser les prédictions
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
    plt.colorbar(scatter)
    plt.xlabel('Age')
    plt.ylabel('Classe')
    plt.title('Prédictions de survie selon l\'âge et la classe')

    # Ajout d'une légende
    plt.legend(*scatter.legend_elements(), title="Survie")
    
    # Affichage du graphique dans Streamlit
    st.pyplot(plt)

    # Nettoyage de la figure matplotlib
    plt.clf()