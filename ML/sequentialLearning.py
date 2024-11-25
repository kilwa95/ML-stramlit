import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model_sequential(model, X_train, y_train, X_test, y_test, model_name):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    # Prédictions
    y_pred = model.predict(X_test)
    # Calcul de l'accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return model_name, accuracy, y_pred

def render_sequential_learning():
    st.title("Apprentissage Ensembliste Séquentiel - Dataset Titanic")
    
    # Chargement des données
    df = pd.read_csv('data/titanic.csv')
    
    # Nettoyage des données
    df.dropna(inplace=True)
    
    # Affichage du dataset
    st.subheader("Aperçu des données")
    st.dataframe(df.head())
    
    # Préparation des features
    X = df[["Age", "Pclass", "Fare"]].values
    y = df["Survived"].values

    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Création des modèles
    models = [
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting"),
        (DecisionTreeClassifier(random_state=42), "Decision Tree")
    ]
    
    # Entraînement séquentiel des modèles
    results = []
    predictions = []
    
    with st.spinner("Entraînement séquentiel des modèles..."):
        for model, name in models:
            st.write(f"Entraînement du modèle: {name}")
            model_name, accuracy, y_pred = train_model_sequential(model, X_train, y_train, X_test, y_test, name)
            results.append((model_name, accuracy))
            predictions.append(y_pred)
            
            # Affichage de la progression
            st.progress(len(results) / len(models))
    
    # Affichage des résultats individuels
    st.subheader("Performances des modèles")
    for name, acc in results:
        st.write(f"{name}: {acc:.3f}")
    
    # Vote majoritaire
    ensemble_predictions = np.array(predictions)
    final_predictions = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(),
        axis=0,
        arr=ensemble_predictions
    )
    
    # Calcul de l'accuracy de l'ensemble
    ensemble_accuracy = accuracy_score(y_test, final_predictions)
    st.write(f"Accuracy de l'ensemble (vote majoritaire): {ensemble_accuracy:.3f}")
    
    # Visualisation des prédictions
    st.subheader("Visualisation des prédictions")
    
    # Création d'un DataFrame pour la visualisation
    results_df = pd.DataFrame({
        'Age': X_test[:, 0],
        'Fare': X_test[:, 2],
        'Predicted': final_predictions,
        'Actual': y_test
    })
    
    # Création du scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = plt.scatter(
        results_df['Age'],
        results_df['Fare'],
        c=results_df['Predicted'],
        cmap='coolwarm',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.xlabel('Age')
    plt.ylabel('Tarif')
    plt.title('Prédictions de survie selon l\'âge et le tarif')
    
    # Affichage du graphique
    st.pyplot(fig)
    
    # Matrice de confusion pour l'ensemble
    st.subheader("Matrice de confusion de l'ensemble")
    conf_matrix = pd.crosstab(results_df['Actual'], results_df['Predicted'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    st.pyplot(fig)
