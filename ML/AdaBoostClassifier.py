import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def render_adaboost_classifier():
    # Titre de la page
    st.title("Classification avec AdaBoost - Dataset Titanic")
    
    # Description
    st.write("""
    Cette démonstration montre comment AdaBoost peut aider à améliorer les performances 
    de classification tout en évitant l'overfitting. Nous allons d'abord voir un exemple 
    d'overfitting avec un arbre de décision simple, puis montrer comment AdaBoost permet 
    de le corriger.
    """)
    
    # Chargement des données
    df = pd.read_csv('data/titanic.csv')
    
    # Nettoyage du jeu de données
    df.dropna(inplace=True)
    
    # Affichage du dataset
    st.subheader("Aperçu des données")
    st.dataframe(df.head())
    
    # Sélection des caractéristiques
    X = df[["Age", "Pclass", "Fare"]].values
    y = df["Survived"].values
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Démonstration de l'overfitting avec un arbre de décision simple
    st.subheader("1. Problème d'overfitting avec un arbre de décision simple")
    
    # Création d'un arbre de décision avec une profondeur importante
    dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_overfit.fit(X_train, y_train)
    
    # Calcul des scores pour montrer l'overfitting
    train_score_dt = dt_overfit.score(X_train, y_train)
    test_score_dt = dt_overfit.score(X_test, y_test)
    
    st.write(f"Score sur l'ensemble d'entraînement: {train_score_dt:.3f}")
    st.write(f"Score sur l'ensemble de test: {test_score_dt:.3f}")
    st.write(f"Différence (overfitting): {train_score_dt - test_score_dt:.3f}")
    
    # 2. Solution avec AdaBoost
    st.subheader("2. Solution avec AdaBoostClassifier")
    
    # Création de l'AdaBoostClassifier
    ada_boost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),  # Arbre de décision peu profond
        n_estimators=100,  # Nombre d'estimateurs
        learning_rate=1.0,  # Taux d'apprentissage
        random_state=42
    )
    
    # Entraînement du modèle
    ada_boost.fit(X_train, y_train)
    
    # Calcul des scores avec AdaBoost
    train_score_ada = ada_boost.score(X_train, y_train)
    test_score_ada = ada_boost.score(X_test, y_test)
    
    st.write(f"Score sur l'ensemble d'entraînement: {train_score_ada:.3f}")
    st.write(f"Score sur l'ensemble de test: {test_score_ada:.3f}")
    st.write(f"Différence: {train_score_ada - test_score_ada:.3f}")
    
    # Comparaison visuelle
    st.subheader("Comparaison des performances")
    
    comparison_data = pd.DataFrame({
        'Modèle': ['Arbre simple', 'Arbre simple', 'AdaBoost', 'AdaBoost'],
        'Ensemble': ['Train', 'Test', 'Train', 'Test'],
        'Score': [train_score_dt, test_score_dt, train_score_ada, test_score_ada]
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comparison_data, x='Modèle', y='Score', hue='Ensemble')
    plt.title('Comparaison des performances entre Arbre simple et AdaBoost')
    st.pyplot(fig)
    
    # Nettoyage de la figure matplotlib
    plt.clf() 