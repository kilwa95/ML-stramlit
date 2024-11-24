import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

def render_logistic_regression():
    data = load_breast_cancer()

    # Titre de la page
    st.title("Cancer du Sein - Classification avec Régression Logistique")
    
    # Description du dataset
    st.write("""
    Ce jeu de données contient des caractéristiques extraites d'images numériques de masses mammaires.
    Les caractéristiques décrivent les propriétés des noyaux cellulaires présents dans l'image.
    
    L'objectif est de classifier les tumeurs comme malignes (cancéreuses) ou bénignes (non cancéreuses)
    en utilisant ces caractéristiques et un modèle de régression logistique.
    
    Le dataset contient 569 échantillons avec 30 caractéristiques différentes pour chaque échantillon.
    """)
    
    st.markdown("---")
    
    # Affichage des données avec un dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    st.write(df)

   

    
    # Définition des variables X (features) et y (target)
    np.random.seed(0)
    X = data.data
    y = data.target
    
    # Division des données en ensembles d'entraînement et de test (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    
    # Création et entraînement du modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Visualisation de la matrice de confusion
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    st.write(conf_matrix)
    sns.heatmap(conf_matrix, annot = True, cmap = "inferno")
    plt.xlabel("Prédictions")
    plt.ylabel("Valeurs réelles")
    st.pyplot(plt)

