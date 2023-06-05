import streamlit as st

from strokePred import dtc, rf, knn, svm
from strokePred import X_test, y_test
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
    button_pressed1 = st.sidebar.button('ROC görbe megjelenítése')
    button_pressed2 = st.sidebar.button('Modellek összevetése')
    st.title("STROKE ELŐREJELZŐ APP")

    if button_pressed1:
        # Tesztadatok előrejelzése
        y_pred = rf.predict(X_test)  # Első oszlopban a pozitív osztály előrejelzéseinek valószínűségeit tároljuk

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
            
            y_pred_dtc = dtc.predict(X_test)
            y_pred_rf = dtc.predict(X_test)
            y_pred_knn = dtc.predict(X_test)
            y_pred_svm = dtc.predict(X_test)
            
            accuracy_dtc = rf.accuracy_score(X_test, y_pred_dtc)
            accuracy_rf = rf.accuracy_score(X_test, y_pred_rf)
            accuracy_knn = knn.accuracy_score(X_test, y_pred_knn)
            accuracy_svm = svm.accuracy_score(X_test, y_pred_svm)
            
            Recall_dtc = rf.recall_score(X_test, y_pred_dtc)
            Recall_rf = rf.recall_score(X_test, y_pred_rf)
            Recall_knn = knn.recall_score(X_test, y_pred_knn)
            Recall_svm = svm.recall_score(X_test, y_pred_svm)
            
            f1_dtc = rf.f1_score(X_test, y_pred_dtc)
            f1_rf = rf.f1_score(X_test, y_pred_rf)
            f1_knn = knn.f1_score(X_test, y_pred_knn)
            f1_svm = svm.f1_score(X_test, y_pred_svm)
           
            # Kiíratás
            st.write('ACCURACY <br>')
            st.write('RandomForest pontossága: {}%'.format(round((accuracy_dtc*100),2)))
            st.write('RandomForest pontossága: {}%'.format(round((accuracy_rf*100),2)))
            st.write('KNN pontossága: {}%'.format(round((accuracy_knn*100),2)))
            st.write('SVM pontossága: {}%'.format(round((accuracy_svm*100),2)))
            
            
            st.write('<br>RECALL')
            st.write('RandomForest pontossága: {}%'.format(round((Recall_dtc*100),2)))
            st.write('RandomForest pontossága: {}%'.format(round((Recall_rf*100),2)))
            st.write('KNN pontossága: {}%'.format(round((Recall_knn*100),2)))
            st.write('SVM pontossága: {}%'.format(round((Recall_svm*100),2)))
            
            st.write('<br>F1')
            st.write('RandomForest pontossága: {}%'.format(round((f1_dtc*100),2)))
            st.write('RandomForest pontossága: {}%'.format(round((f1_rf*100),2)))
            st.write('KNN pontossága: {}%'.format(round((f1_knn*100),2)))
            st.write('SVM pontossága: {}%'.format(round((f1_svm*100),2)))
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
