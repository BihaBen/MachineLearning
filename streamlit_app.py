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
    st.title("STROKE ELŐREJELZŐ APP")
    
    if button_pressed1:
        y_predict_rf_matrix = rf1.predict(X_test)
        cm = confusion_matrix(y_test, y_predict_rf_matrix)

        # Confusion matrix megjelenítése
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)
        
        recall = recall_score(y_test, y_predict_rf_matrix)
        #precision = precision_score(y_test, y_predict_rf_matrix, pos_label=1)
        #f1 = f1_score(y_test, y_predict_rf_matrix, pos_label=1)
        accuracy = rf1.score(y_test, y_predict_rf_matrix)
        st.write('RandomForest accuracy:', accuracy)
        st.write('RandomForest recall:', recall)
        #st.write('RandomForest precision:', precision)
        #st.write('RandomForest F1 score:', f1)
        
    if button_pressed2:
        with st.sidebar:
            y_pred_dtc = dtc1.predict(X_test)
            y_pred_rf = rf1.predict(X_test)
            y_pred_knn = knn1.predict(X_test)
            y_pred_svm = svm1.predict(X_test)
            
            accuracy_dtc = dtc1.accuracy_score(X_test, y_pred_dtc)
            accuracy_rf = rf1.accuracy_score(X_test, y_pred_rf)
            accuracy_knn = knn1.accuracy_score(X_test, y_pred_knn)
            accuracy_svm = svm1.accuracy_score(X_test, y_pred_svm)
            
            Recall_dtc = dtc1.recall_score(X_test, y_pred_dtc)
            Recall_rf = rf1.recall_score(X_test, y_pred_rf)
            Recall_knn = knn1.recall_score(X_test, y_pred_knn)
            Recall_svm = svm1.recall_score(X_test, y_pred_svm)
            
            f1_dtc = dtc1.f1_score(X_test, y_pred_dtc)
            f1_rf = rf1.f1_score(X_test, y_pred_rf)
            f1_knn = knn1.f1_score(X_test, y_pred_knn)
            f1_svm = svm1.f1_score(X_test, y_pred_svm)
           
            # Kiíratás
            st.write('ACCURACY')
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
