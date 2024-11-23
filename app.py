import streamlit as st
from components.header import render_header
from components.sidebar import render_sidebar
from components.charts import render_charts
from components.tables import render_tables
from components.images import render_images
from components.videos import render_videos
from components.buttons import render_buttons
from components.sliders import render_sliders
from components.selectors import render_selectors
from components.forms import render_forms
from components.ml_models import render_ml_models
from components.theme import render_theme_customization
from components.data_viewer import render_data_viewer

def main():
    render_header("Mon Application Streamlit")
    selected = render_sidebar()
    
    if selected == "Analyses":
        render_charts()
    elif selected == "Données":
        render_tables()
    elif selected == "Images":
        render_images()
    elif selected == "Vidéos":
        render_videos()
    elif selected == "Widgets":
        widget_type = st.selectbox(
            "Choisissez le type de widget",
            ["Boutons", "Sliders", "Sélecteurs"]
        )
        
        if widget_type == "Boutons":
            render_buttons()
        elif widget_type == "Sliders":
            render_sliders()
        elif widget_type == "Sélecteurs":
            render_selectors()
    elif selected == "Formulaires":
        render_forms()
    elif selected == "Machine Learning":
        render_ml_models()
    elif selected == "Thème":
        render_theme_customization()
    elif selected == "Base de données":
        render_data_viewer()
    else:
        st.write("Page d'accueil")

if __name__ == "__main__":
    main() 