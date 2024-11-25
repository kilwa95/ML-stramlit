import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("Menu")
        st.write("Bienvenue dans l'application!")
        
        selected = st.radio(
            "Navigation",
            ["Accueil","LogisticRegression","KNN","DecisionTreeClassifier","ParallelLearning","SequentialLearning","RandomForestClassifier","BaggingClassifier","AdaBoostClassifier"]
        )
        
    return selected 