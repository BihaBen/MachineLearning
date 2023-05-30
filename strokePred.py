#import
import pandas as pd
import numpy as np

#Döntési fa
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#Random forest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import *
#KNN
from sklearn.neighbors import KNeighborsClassifier
#SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#Plot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#A csv tartalmának pandas dataframe-be olvasása
file = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = file

#------------------------------------------------------------------
# TANÍTÁS
#------------------------------------------------------------------
# Feature-ökre és labelekre felbontás
#X = df.iloc[:, :11] # 1.oszloptól 11.-ig és minden sor
#Y = df.iloc[:, 11] # 12. oszlop(eredmény) minden sora
X = df.drop(columns=['stroke'])
Y= df['stroke'].values
# X-> Feature Y-> Test
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=73)

# Teszt
#print("A tanító adatok száma: " + str(X_train.shape))
#print("A tanító adatok célváltozóinak száma: " + str(Y_train.shape))
#print("\nA predikáló adatok száma: " + str(x_test.shape))
#print("A predikáló célváltozóinak száma: " + str(y_test.shape))

#------------------------------------------------------------------
# DÖNTÉSI FA
#------------------------------------------------------------------

# Döntési fa osztályozó objectum készítése
dtc = DecisionTreeClassifier()

# Döntési fa feltanítása tanító adatokkal
dtc.fit(X_train, Y_train)

# Eredmények ellenőrzése
accuracy = dtc.score(x_test, y_test)

# Pontosság kiírása
print("Accuracy:", accuracy)

#------------------------------------------------------------------
# DÖNTÉSI FA
#------------------------------------------------------------------

# Random Forest készítése: 100db fa behelyezés
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Tanító adatokkal feltanítom a modelt
rf.fit(X_train, Y_train)

# Eredmények ellenőrzése
accuracy = rf.score(x_test, y_test)

# Pontosság kiírása
print("Accuracy:", accuracy)

#------------------------------------------------------------------
# KNN
#------------------------------------------------------------------

# KNN osztályozó készítése 5 megengedett szomszéddal
knn = KNeighborsClassifier(n_neighbors=5)

# Tanító adatokkal feltanítom a modelt
knn.fit(X_train, Y_train)

# Eredmények ellenőrzése
accuracy = knn.score(x_test, y_test)

# Pontosság kiírása
print("Accuracy:", accuracy)

#------------------------------------------------------------------
# SVM
#------------------------------------------------------------------

# SVM modell illesztése
# A modell lineáris határvonalakkal választja el az osztályokat
# A C paraméter a szabályozási paraméter, amely befolyásolja az SVM modell kompromisszumát a túltanulás és az alultanulás között. Minél nagyobb a C, annál kevésbé tolerálja az SVM a hibákat a döntési határon.
svm = SVC(kernel='linear', C=1.0)

# Tanító adatokkal feltanítom a modelt
svm.fit(X_train, Y_train)

# Eredmények ellenőrzése
accuracy = accuracy_score(y_test, svm.predict(x_test))

# Pontosság kiírása
print(f"Model accuracy: {accuracy}")
