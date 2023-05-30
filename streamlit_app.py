import streamlit as st

from strokePred import rf,knn,dtc
from strokePred import x_test, y_test

from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix

def main():
    st.title('Stroke előrejelző app')

    if st.button('Konfúziós mátrix megjelenítése'):
        cm = confusion_matrix(y_test, knn.predict(x_test))
        # Megjelenítés
        fig, ax = plot_confusion_matrix(conf_mat=cm)
        st.pyplot(fig)

    if st.button('Modellek összevetése'):
        rf_accuracy = rf.score(x_test, y_test)
        knn_accuracy = knn.score(x_test, y_test)
        dtc_accuracy = dtc.score(x_test, y_test)
        # Kiíratás
        st.write('RandomForest pontossága: {}%'.format(rf_accuracy*100))
        st.write('KNN pontossága:{}%'.format(knn_accuracy*100))
        st.write('SVM pontossága:{}%'.format(dtc_accuracy*100))

if __name__ == '__main__':
    main()
