import streamlit as st

from strokePred import dtc, rf, knn, svm
from strokePred import X_test, y_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

#Metrikák
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
    st.title("STROKE ELŐREJELZŐ APP")
    
    if button_pressed1:
         cm = confusion_matrix(y_test, rf.predict(X_test))

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

        recall = recall_score(y_test, model.predict(X_test), pos_label=1)
        precision = precision_score(y_test, model.predict(X_test), pos_label=1)
        f1 = f1_score(y_test, model.predict(X_test), pos_label=1)
        accuracy = model.score(X_test, y_test)
        st.write('RandomForest accuracy:', accuracy)
        st.write('RandomForest recall:', recall)
        st.write('RandomForest precision:', precision)
        st.write('RandomForest F1 score:', f1)
        
    if button_pressed2:
        with st.sidebar:
           y_pred_rf = dtc.predict(X_test)
           st.write(rf.accuracy_score(X_test, y_pred_dtc))
            
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
