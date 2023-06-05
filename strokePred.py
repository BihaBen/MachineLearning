#import
import pandas as pd
import numpy as np
# Tanítás
from sklearn.model_selection import train_test_split, GridSearchCV
# Döntési fa
from sklearn.tree import DecisionTreeClassifier
# Random forest
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.datasets import make_classification
# KNN
from sklearn.neighbors import KNeighborsClassifier
# SVM
from sklearn.svm import SVC
#Plot
import matplotlib.pyplot as plt
#Metrikák
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#Adatnormalizalas
from sklearn.preprocessing import PowerTransformer 
#Konfuzios matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#A csv tartalmának pandas dataframe-be olvasása
file = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = file
df.head

#------------------------------------------------------------------
#TESZT
#------------------------------------------------------------------
print('TESZTELÉS EREDMÉNYEI:')
print(f'\nAz adatok mennyisége = {df.shape}')
print(f'\nTöbbször előforduló adat = {df.duplicated().sum()}')
print(f'\nTisztítás előtti üres cellák oszloponként:\n {df.isnull().sum()}')
print(f'\nOszlop statisztika szemléltetése:')
print(f' {round(df.describe(),2)}')
print(f'\nElőforduló nemek:{df.gender.unique()}\n')
num_zeros = sum(df['stroke'] == 0)
num_ones = sum(df['stroke'] == 1)
# Eredmények kiírása
print(f"A y_train változóban {num_zeros} darab nulla és {num_ones} darab egyes érték van.")

#------------------------------------------------------------------
# TISZTÍTÁS
#------------------------------------------------------------------
print('\nTISZTÍTÁS EREDMÉNYEI:')

# NULL típusú cellák sorainak eldobása
df.dropna(axis=0, inplace=True)

# átlagos cukorszint konvertálása -> Magyar értékké konvertálás
df.avg_glucose_level = [round(x / 18.016,2) for x in df.avg_glucose_level]

# Számok tartalmazó oszlopok kerekítése 0 tizedesjegyre
numeric_columns = df.select_dtypes(include=[float, int]).columns
df[numeric_columns] = df[(numeric_columns)].astype(int)

# nemek konvertálása
#print(f'\nAz egyéb nemet férfi nemre cserélem:{df.gender.unique()}')
#df['gender']=df['gender'].replace({'Other': 'Male'})
#df['gender'] = df['gender'].replace({'Female': 0 ,'Male': 1})


# dohányzási szokás konvertálás
df = df[df['smoking_status'] != 'Unknown']
df.loc[df['smoking_status'] == 'smokes', 'smoking_status'] = 0
df.loc[df['smoking_status'] == 'formerly smoked', 'smoking_status'] = 1
df.loc[df['smoking_status'] == 'never smoked', 'smoking_status'] = 2

# Felesleges oszlopok eldobása
df =df.drop(['gender','age','work_type','id', 'ever_married','Residence_type'], axis=1)
print(f'\nA maradt oszlopok nevei:\n{df.columns}')
print(f'\nOszlop eldobások után maradandó dimenzió={df.shape}')

print(f'\nFennmaradó üres cellák száma:\n{df.isnull().sum()}\n')


# Sorok eldobása, ahol a 'BMI' oszlop értéke nem esik 7 és 40 közé
df = df[(df['bmi'] >= 7) & (df['bmi'] <= 40)]


#smoking -> Számomra hihetetlen, ha 14 év alatt cigizik az illető és még nyilatkozik is róla
#df = df.loc[~((file['age'] < 14) & (df['smoking_status'] == 'smokes')) | ((df['age'] < 14) & (df['smoking_status'] == 'formerly smoked'))]

# STROKE-osok kellenek a valós stroke-osok megmutatásához és betöltök hozzá hasonló mennyiségű egészséges embert
df_old = df
df_new = df[df['stroke'] == 1]
num_have_stroke = len(df_new)
df_new = pd.concat([df_new, df.iloc[num_have_stroke::23]])
df = df_new

print(f'\nAdatok szemléltetése:')
print(df.head(5))

print(f'\nOszlop statisztika szemléltetése:')
print(f' {round(df.describe(),2)}')
#------------------------------------------------------------------
# TANÍTÁS
#------------------------------------------------------------------
print('\n\nTANÍTÁS EREDMÉNYEI:')
# Feature-ökre és labelekre felbontás
#X = df.iloc[:, :11] # 1.oszloptól 11.-ig és minden sor
#Y = df.iloc[:, 11] # 12. oszlop(eredmény) minden sora

#-----------------------------Nagy dataFrame felosztás (teszthez)-----------------------------------
XX = df_old.drop('stroke', axis=1)
scaler = PowerTransformer()
XX = scaler.fit_transform(XX)
yy= df_old['stroke'].values
XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size=0.4, random_state=42)
#--------------------------------------------------------------------------------------------------

X = df.drop('stroke', axis=1)
scaler = PowerTransformer()
X = scaler.fit_transform(X)
y= df['stroke'].values
# X-> Feature Y-> Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Nullák és egyesek megszámlálása a célváltozóban
num_zeros = sum(y_test == 0)
num_ones = sum(y_test == 1)
# Eredmények kiírása
print(f'\nA y_train változóban {num_zeros} darab nulla és {num_ones} darab egyes érték van.\n')

#------------------------------------------------------------------
# DÖNTÉSI FA (Kis adattal tanítás kis adattal tesztelés)
#------------------------------------------------------------------

# Döntési fa osztályozó objectum készítése
dtc = DecisionTreeClassifier(max_leaf_nodes= 25 , min_samples_split=3, min_samples_leaf=10)

# Döntési fa feltanítása tanító adatokkal
dtc.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
y_pred = dtc.predict(X_test)

# Pontosság kiszámítása
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("[Decision tree  -> kis modell kis teszt] Accuracy pontosság: ", accuracy)
print("[Decision tree  -> kis modell kis teszt] Recall pontosság: ", recall)
print("[Decision tree  -> kis modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(y_test, y_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# DÖNTÉSI FA (Kis adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------

# Döntési fa osztályozó objectum készítése
#dtc = DecisionTreeClassifier(max_leaf_nodes= 25 , min_samples_split=3, min_samples_leaf=10)

# Döntési fa feltanítása tanító adatokkal
#dtc.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
yy_pred = dtc.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[Decision tree  -> kis modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[Decision tree  -> kis modell nagy teszt] Recall pontosság: ", recall)
print("[Decision tree  -> kis modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# DÖNTÉSI FA (Nagy adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------

# Döntési fa osztályozó objectum készítése
#dtc = DecisionTreeClassifier(max_leaf_nodes= 25 , min_samples_split=3, min_samples_leaf=10)

# Döntési fa feltanítása tanító adatokkal
dtc.fit(XX_train, yy_train)

# Előrejelzés a teszt adatokon
yy_pred = dtc.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[Decision tree  -> nagy modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[Decision tree  -> nagy modell nagy teszt] Recall pontosság: ", recall)
print("[Decision tree  -> nagy modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()
#------------------------------------------------------------------
# RANDOM FOREST (Kis adattal tanítás kis adattal tesztelés)
#------------------------------------------------------------------
# Random Forest inicializálása
rf = RandomForestClassifier(n_estimators=100, random_state=40, max_depth=2, min_samples_split=30, min_samples_leaf=15)

# Tanítás
rf.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
y_pred = rf.predict(X_test)

# Pontosság kiszámítása
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("[Random forest] Accuracy pontosság: ", accuracy)
print("[Random forest] Recall pontosság: ", recall)
print("[Random forest] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(y_test, y_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# RANDOM FOREST (Kis adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------
# Random Forest inicializálása
#rf = RandomForestClassifier(n_estimators=100, random_state=40, max_depth=2, min_samples_split=30, min_samples_leaf=15)

# Tanítás
rf.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
yy_pred = rf.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[Random forest -> kis modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[Random forest -> kis modell nagy teszt] Recall pontosság: ", recall)
print("[Random forest -> kis modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# RANDOM FOREST (Nagy adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------
# Random Forest inicializálása
#rf = RandomForestClassifier(n_estimators=100, random_state=40, max_depth=2, min_samples_split=30, min_samples_leaf=15)

# Tanítás
rf.fit(XX_train, yy_train)

# Előrejelzés a teszt adatokon
yy_pred = rf.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[Random forest -> nagy modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[Random forest -> nagy modell nagy teszt] Recall pontosság: ", recall)
print("[Random forest -> nagy modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# KNN (Kis adattal tanítás kis adattal tesztelés)
#------------------------------------------------------------------

# KNN osztályozó készítése 5 megengedett szomszéddal
knn = KNeighborsClassifier(n_neighbors=13)

# Tanító adatokkal feltanítom a modelt
knn.fit(X_train, y_train)
# Előrejelzés a teszt adatokon
y_pred = knn.predict(X_test)

# Pontosság kiszámítása
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("[KNN -> kis modell kis teszt] Accuracy pontosság: ", accuracy)
print("[KNN -> kis modell kis teszt] Recall pontosság: ", recall)
print("[KNN -> kis modell kis teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(y_test, y_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# KNN (Kis adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------

# KNN osztályozó készítése 5 megengedett szomszéddal
#knn = KNeighborsClassifier(n_neighbors=13)

# Tanító adatokkal feltanítom a modelt
knn.fit(X_train, y_train)
# Előrejelzés a teszt adatokon
yy_pred = knn.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[KNN -> kis modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[KNN -> kis modell nagy teszt] Recall pontosság: ", recall)
print("[KNN -> kis modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# KNN (Nagy adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------

# KNN osztályozó készítése 5 megengedett szomszéddal
#knn = KNeighborsClassifier(n_neighbors=13)

# Tanító adatokkal feltanítom a modelt
knn.fit(XX_train, yy_train)
# Előrejelzés a teszt adatokon
yy_pred = knn.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[KNN -> nagy modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[KNN -> nagy modell nagy teszt] Recall pontosság: ", recall)
print("[KNN -> nagy modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# SVM (Kis adattal tanítás kis adattal tesztelés)
#------------------------------------------------------------------


# SVM modell illesztése
# A modell lineáris határvonalakkal választja el az osztályokat
# A C paraméter a szabályozási paraméter, amely befolyásolja az SVM modell kompromisszumát a túltanulás és az alultanulás között. Minél nagyobb a C, annál kevésbé tolerálja az SVM a hibákat a döntési határon.
svm = SVC(kernel='linear', C=1.0)

# Tanító adatokkal feltanítom a modelt
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Pontosság kiszámítása
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("[SVM -> kis modell kis teszt] Accuracy pontosság: ", accuracy)
print("[SVM -> kis modell kis teszt] Recall pontosság: ", recall)
print("[SVM -> kis modell kis teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(y_test, y_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# SVM (Kis adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------


# SVM modell illesztése
# A modell lineáris határvonalakkal választja el az osztályokat
# A C paraméter a szabályozási paraméter, amely befolyásolja az SVM modell kompromisszumát a túltanulás és az alultanulás között. Minél nagyobb a C, annál kevésbé tolerálja az SVM a hibákat a döntési határon.
#svm = SVC(kernel='linear', C=1.0)

# Tanító adatokkal feltanítom a modelt
svm.fit(X_train, y_train)
yy_pred = svm.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[SVM -> kis modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[SVM -> kis modell nagy teszt] Recall pontosság: ", recall)
print("[SVM -> kis modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()

#------------------------------------------------------------------
# SVM (Nagy adattal tanítás nagy adattal tesztelés)
#------------------------------------------------------------------


# SVM modell illesztése
# A modell lineáris határvonalakkal választja el az osztályokat
# A C paraméter a szabályozási paraméter, amely befolyásolja az SVM modell kompromisszumát a túltanulás és az alultanulás között. Minél nagyobb a C, annál kevésbé tolerálja az SVM a hibákat a döntési határon.
#svm = SVC(kernel='linear', C=1.0)

# Tanító adatokkal feltanítom a modelt
svm.fit(XX_train, yy_train)
yy_pred = svm.predict(XX_test)

# Pontosság kiszámítása
accuracy = accuracy_score(yy_test, yy_pred)
recall = recall_score(yy_test, yy_pred)
f1 = f1_score(yy_test, yy_pred)
print("[SVM -> nagy modell nagy teszt] Accuracy pontosság: ", accuracy)
print("[SVM -> nagy modell nagy teszt] Recall pontosság: ", recall)
print("[SVM-> nagy modell nagy teszt] F1 pontosság: ", f1)

# Konfúziós mátrix létrehozása
cm = confusion_matrix(yy_test, yy_pred)
# Konfúziós mátrix megjelenítése
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Előrejelzett osztály')
plt.ylabel('Valós osztály')
plt.show()
