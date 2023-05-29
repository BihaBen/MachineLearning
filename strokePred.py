import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#KNN
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn import svm

#-------------------------------------------------------
file=pd.read_csv('Covid Data.csv')
df=file.copy()


#-------------------------------------------------------
X=df.drop('stroke',axis=1).values
Y=df['stroke'].values
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=73)
#-------------------------------------------------------
# Random Forest készítése: 100db fa behelyezés
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Tanító adatokkal feltanítom a modelt
rf.fit(X_train, Y_train)

# A "predict" metódus meghívásával a modell előrejelzést készít a teszthalmazon lévő adatpontokra
y_pred = rf.predict(x_test)

# Eredmények ellenőrzése
accuracy = rf.score(y_pred, y_test)

# Pontosság kiírása
print("Accuracy:", accuracy)
#-------------------------------------------------------
# KNN osztályozó készítése 5 megengedett szomszéddal
knn = KNeighborsClassifier(n_neighbors=5)

# Tanító adatokkal feltanítom a modelt
knn.fit(X_train, Y_train)

# A "predict" metódus meghívásával a modell előrejelzést készít a teszthalmazon lévő adatpontokra
y_pred = knn.predict(x_test)

# Eredmények ellenőrzése
accuracy = knn.score(y_pred, y_test)

# Pontosság kiírása
print("Accuracy:", accuracy)
#-------------------------------------------------------
# SVM modell illesztése
# A modell lineáris határvonalakkal választja el az osztályokat
# A C paraméter a szabályozási paraméter, amely befolyásolja az SVM modell kompromisszumát a túltanulás és az alultanulás között. Minél nagyobb a C, annál kevésbé tolerálja az SVM a hibákat a döntési határon.
svm = svm.SVC(kernel='linear', C=1.0)

# Tanító adatokkal feltanítom a modelt
svm.fit(X_train, Y_train)

# A "predict" metódus meghívásával a modell előrejelzést készít a teszthalmazon lévő adatpontokra
y_pred = svm.predict(x_test)

# Eredmények ellenőrzése
accuracy = accuracy_score(y_pred, y_test)

# Pontosság kiírása
print(f"Model accuracy: {accuracy}")







