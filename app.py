import streamlit as st
from components.header import render_header
from components.sidebar import render_sidebar
from ML.LogisticRegression import render_logistic_regression
from ML.KNN import render_knn
from ML.DecisionTreeClassifier import render_decision_tree_classifier
from ML.parallelLearning import render_parallel_learning
from ML.sequentialLearning import render_sequential_learning
from ML.BaggingClassifier import render_bagging_classifier
from ML.AdaBoostClassifier import render_adaboost_classifier
def main():
    render_header("Mon Application Streamlit")
    selected = render_sidebar()

    if selected == "LogisticRegression":
        render_logistic_regression()

    if selected == "KNN":
        render_knn()
    
    if selected == "DecisionTreeClassifier":
        render_decision_tree_classifier()
    
    if selected == "ParallelLearning":
        render_parallel_learning()
    
    if selected == "SequentialLearning":
        render_sequential_learning()

    
    if selected == "BaggingClassifier":
        render_bagging_classifier()
    
    if selected == "AdaBoostClassifier":
        render_adaboost_classifier()
  
if __name__ == "__main__":
    main() 