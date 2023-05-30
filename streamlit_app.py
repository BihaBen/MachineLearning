import streamlit as st

from strokePred import rf,knn,dtc, svm
from strokePred import x_test, y_test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    button_pressed1 = st.sidebar.button('ROC görbe megjelenítése')
    button_pressed2 = st.sidebar.button('Modellek összevetése')
    st.title("STROKE ELŐREJELZŐ APP")

    if button_pressed1:
        # Tesztadatok előrejelzése
        y_pred = rf.predict_proba(x_test)[:, 1]  # Első oszlopban a pozitív osztály előrejelzéseinek valószínűségeit tároljuk

        # ROC görbe számítása
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Streamlit alkalmazás
        st.title("ROC görbe")
        # ROC görbe megjelenítése
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC görbe (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('True Positive arány')
        plt.ylabel('False Positive arány')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)
    if button_pressed2:
        with st.sidebar:
            rf_accuracy = rf.score(x_test, y_test)
            knn_accuracy = knn.score(x_test, y_test)
            svm_accuracy = svm.score(x_test, y_test)
            # Kiíratás
            st.write('RandomForest pontossága: {}%'.format(round((rf_accuracy*100),2)))
            st.write('KNN pontossága: {}%'.format(round((knn_accuracy*100),2)))
            st.write('SVM pontossága: {}%'.format(round((svm_accuracy*100),2)))
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
