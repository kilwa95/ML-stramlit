import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def render_bagging_classifier():
    # Titre de la page
    st.title("Classification avec Bagging - Dataset Titanic")
    
    # Description
    st.write("""
    Cette démonstration montre comment le Bagging peut aider à réduire l'overfitting.
    Nous allons d'abord voir un exemple d'overfitting avec un arbre de décision simple,
    puis montrer comment le Bagging permet de le corriger.
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
    
    # 2. Solution avec Bagging
    st.subheader("2. Solution avec BaggingClassifier")
    
    # Création du BaggingClassifier
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        random_state=42
    )
    
    # Entraînement du modèle
    bagging.fit(X_train, y_train)
    
    # Calcul des scores avec Bagging
    train_score_bagging = bagging.score(X_train, y_train)
    test_score_bagging = bagging.score(X_test, y_test)
    
    st.write(f"Score sur l'ensemble d'entraînement: {train_score_bagging:.3f}")
    st.write(f"Score sur l'ensemble de test: {test_score_bagging:.3f}")
    st.write(f"Différence: {train_score_bagging - test_score_bagging:.3f}")
    
    # Comparaison visuelle
    st.subheader("Comparaison des performances")
    
    comparison_data = pd.DataFrame({
        'Modèle': ['Arbre simple', 'Arbre simple', 'Bagging', 'Bagging'],
        'Ensemble': ['Train', 'Test', 'Train', 'Test'],
        'Score': [train_score_dt, test_score_dt, train_score_bagging, test_score_bagging]
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comparison_data, x='Modèle', y='Score', hue='Ensemble')
    plt.title('Comparaison des performances entre Arbre simple et Bagging')
    st.pyplot(fig)
    
    # Nettoyage de la figure matplotlib
    plt.clf()
