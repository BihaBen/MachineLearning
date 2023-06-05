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
        y_pred_rf = dtc.predict(X_test)
        # Konfúziós mátrix létrehozása
        cm = confusion_matrix(X_test, y_pred)

        # Streamlit alkalmazás
        st.title('Konfúziós mátrix')
        st.write('Ez egy példa a konfúziós mátrix megjelenítésére a Streamlit segítségével.')

        # Konfúziós mátrix megjelenítése
        df_cm = pd.DataFrame(cm, index=['Valós Negatív', 'Valós Pozitív'], columns=['Prediktált Negatív', 'Prediktált Pozitív'])
        plt.figure(figsize=(6, 4))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        st.pyplot()

        # Pontosság, érzékenység és specificitás kiszámítása
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        st.write('Pontosság:', accuracy)
        st.write('Érzékenység:', sensitivity)
        st.write('Specificitás:', specificity)
        
    if button_pressed2:
        with st.sidebar:
           y_pred_rf = dtc.predict(X_test)
           st.write(rf.accuracy_score(X_test, y_pred_dtc))
            
       

if __name__ == '__main__':
   main()
   add_bg_from_url()
