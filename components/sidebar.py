import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.header("Menu")
        st.write("Bienvenue dans l'application!")
        
        selected = st.radio(
            "Navigation",
            ["Accueil", "Données", "Analyses", "Images", "Vidéos", "Widgets", "Formulaires", "Machine Learning", "Thème"]
        )
        
    return selected 