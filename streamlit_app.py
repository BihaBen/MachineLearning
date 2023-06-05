import streamlit as st

from strokePred import dtc, rf, knn, svm
from strokePred import X_test, y_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

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
        # Tesztadatok előrejelzése
        y_pred = rf.predict(X_test)  # Első oszlopban a pozitív osztály előrejelzéseinek valószínűségeit tároljuk
        # Konfúziós mátrix létrehozása
        cm = confusion_matrix(y_test, y_pred)
        # Konfúziós mátrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)
        
    if button_pressed2:
        with st.sidebar:
            
            y_pred_dtc = dtc.predict(X_test)
            y_pred_rf = dtc.predict(X_test)
            y_pred_knn = dtc.predict(X_test)
            y_pred_svm = dtc.predict(X_test)
            
           
           
            # Kiíratás
            st.write('ACCURACY')
            st.write('RandomForest pontossága: {}%'.format(round((accuracy_dtc*100),2)))
            st.write('RandomForest pontossága: {}%'.format(round((accuracy_rf*100),2)))
            st.write('KNN pontossága: {}%'.format(round((accuracy_knn*100),2)))
            st.write('SVM pontossága: {}%'.format(round((accuracy_svm*100),2)))
            
            
            
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
