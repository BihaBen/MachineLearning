import streamlit as st

from strokePred import dtc1, rf1, knn1, svm1
from strokePred import X_test, y_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

#Metrikák
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(https://static.toiimg.com/photo/msid-87343087/87343087.jpg);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main():
    button_pressed1 = st.sidebar.button('Konfuzios matrix megjelenítése')
    button_pressed2 = st.sidebar.button('Modellek összevetése')
    
    st.markdown('<h1 style="color: white; font-size: 20px;">stroke előrejelző app</h1>', unsafe_allow_html=True)
    
    if button_pressed1:
        y_predict_rf_matrix = rf1.predict(X_test)
        cm = confusion_matrix(y_test, y_predict_rf_matrix)

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        
        ax.set(title='Random forest konfutios matrix szemleltetes:', ylabel='Valós osztály', xlabel='Előrejelzett osztály')
        
        st.pyplot(fig)
        
        
        st.markdown('<h2 style="background-color: white;text-align: center;">Accuracy:</h2>', unsafe_allow_html=True)
        st.write('<h3 style="background-color: white;text-align: center;">(TN + TP) / (TN + FP + TP + FN)</h3>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="background-color: white;text-align: center;">Precision:</h2>', unsafe_allow_html=True)
        st.write('<h3 style="background-color: white;text-align: center;">(TP) / (FP + TP)</h3>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="background-color: white;text-align: center;">Recall:</h2>', unsafe_allow_html=True)
        st.write('<h3 style="background-color: white;text-align: center;">(TP) / (TP + FN):</h3>', unsafe_allow_html=True)
        
        st.markdown('<h2 style="background-color: white;text-align: center;">F1 score:</h2>', unsafe_allow_html=True)
        st.write('<h3 style="background-color: white;text-align: center;">(PRECISION * RECALL) / (PRECISION + RECALL)</h3>', unsafe_allow_html=True)

        
    if button_pressed2:
        with st.sidebar:
            y_predict_dtc = dtc1.predict(X_test)
            y_predict_rf = rf1.predict(X_test)
            y_predict_knn = knn1.predict(X_test)
            y_predict_svm = svm1.predict(X_test)

            accuracy_dtc = accuracy_score(y_test, y_predict_dtc)
            accuracy_rf = accuracy_score(y_test, y_predict_rf)
            accuracy_knn = accuracy_score(y_test, y_predict_knn)
            accuracy_svm = accuracy_score(y_test, y_predict_svm)

            recall_dtc = recall_score(y_test, y_predict_dtc)
            recall_rf = recall_score(y_test, y_predict_rf)
            recall_knn = recall_score(y_test, y_predict_knn)
            recall_svm = recall_score(y_test, y_predict_svm)

            precision_dtc = precision_score(y_test, y_predict_dtc)
            precision_rf = precision_score(y_test, y_predict_rf)
            precision_knn = precision_score(y_test, y_predict_knn)
            precision_svm = precision_score(y_test, y_predict_svm)

            f1_dtc = f1_score(y_test, y_predict_dtc)
            f1_rf = f1_score(y_test, y_predict_rf)
            f1_knn = f1_score(y_test, y_predict_knn)
            f1_svm = f1_score(y_test, y_predict_svm)

            st.markdown("<h3>Decision Tree:</h3>", unsafe_allow_html=True)
            st.write("Accuracy:", accuracy_dtc)
            st.write("Recall:", recall_dtc)
            st.write("Precision:", precision_dtc)
            st.write("F1 Score:", f1_dtc)

            st.markdown("<h3>Random Forest:</h3>", unsafe_allow_html=True)
            st.write("Accuracy:", accuracy_rf)
            st.write("Recall:", recall_rf)
            st.write("Precision:", precision_rf)
            st.write("F1 Score:", f1_rf)

            st.markdown("<h3>KNN:</h3>", unsafe_allow_html=True)
            st.write("Accuracy:", accuracy_knn)
            st.write("Recall:", recall_knn)
            st.write("Precision:", precision_knn)
            st.write("F1 Score:", f1_knn)

            st.markdown("<h3>SVM:</h3>", unsafe_allow_html=True)
            st.write("Accuracy:", accuracy_svm)
            st.write("Recall:", recall_svm)
            st.write("Precision:", precision_svm)
            st.write("F1 Score:", f1_svm)
            
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
