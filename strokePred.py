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
#TESZT
#------------------------------------------------------------------
# Beolvasás tesztelése
print("\nAz adatok sorinak száma:" + " " + str(len(df.index)))
print("Legnagyobb életkor")
print((df["age"].max()))
print("Legkissebb életkor (létezik pár hetes baba)")
print((df["age"].min()))
print("Átlagos életkor")
print((df["age"].mean()))

print("\nLegnagyobb BMI")
print((df["bmi"].max()))
print("Legkissebb BMI")
print((df["bmi"].min()))
print("Átlagos BMI")
print((df["bmi"].mean()))

print("\nLegnagyobb cukorszint")
print((df["avg_glucose_level"].max()))
print("Legkissebb cukorszint")
print((df["avg_glucose_level"].min()))
print("Átlagos cukorszint")
print((df["avg_glucose_level"].mean()))

#------------------------------------------------------------------
# TISZTÍTÁS
#------------------------------------------------------------------
# Minden életkor legyen float
df['age'] = df[["age"]].fillna(0).astype(float) # Létezik integer és float típusú 

# NULL tipusok cserélése Numpy tipusu NULL-ra
df.replace('Nan',np.nan,inplace=True)
df.replace('nan',np.nan,inplace=True)
df.replace('N/A',np.nan,inplace=True)

# Mennyi NULL van a DataSet-ben
print(df.isna().sum())

print("\nAz adatok sorinak száma:" + " " + str(len(df.index)))

# Férfi = 1 Nő = 0
df["gender"] = df["gender"].replace({"Female": 0 ,"Male": 1, "Other": 2})

# Valaha megházasodott-> igen = 1 nem = 0
df["ever_married"] = df["ever_married"].replace({"Yes": 1, "No": 0})

# Városban vagy vidéken él-> városban = 1 vidéken = 0
df["Residence_type"] = df["Residence_type"].replace({"Urban": 1, "Rural": 0})

# Dohányzási szokások számmá alakítás -> "soha": 0, "néha": 1, "dohányzik": 2, "ismeretlen": 3
df["smoking_status"] = df["smoking_status"].replace({"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3 })

# Munkavégzés tipusa dinamikusan egy dictionary-be teszi a kulcs-érték párokat
workType_list = df["work_type"].tolist()
workType_map = {work_type: i for i, work_type in enumerate(df["work_type"].unique())}
df["work_type"] = df["work_type"].replace(workType_map)
workNumb_list = df["work_type"].tolist()
my_dict = dict(zip(workType_list, workNumb_list))
my_dict = {k: v for k, v in set(my_dict.items())}

old_rowN=len(df.index)
#avg_glucose_level -> Magyar értékké konvertálás
df.avg_glucose_level = [round(x / 18.016,2) for x in df.avg_glucose_level]

#bmi -> Amerikai és Magyar számítási különbségek vannak, de az eredmény megegyezik [7 és 40 közötti BMI]
df = df[(df['bmi']> 7) & (df['bmi'] < 40)]

#smoking -> Számomra hihetetlen, ha 10-12 évesen cigizik az illető és még nyilatkozik is róla
df = df.loc[~((file['age'] < 10) & (df['smoking_status'] == 'smokes')) | ((df['age'] < 10) & (df['smoking_status'] == 'formerly smoked'))]

# Adatok sorainak csökkenése
print("\nTörölt adatok száma:" + " " + str(old_rowN - len(df.index)))
print("Maradt még:" + " " + str(len(df.index)))

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
print("A tanító adatok száma: " + str(X_train.shape))
print("A tanító adatok célváltozóinak száma: " + str(Y_train.shape))
print("\nA predikáló adatok száma: " + str(x_test.shape))
print("A predikáló célváltozóinak száma: " + str(y_test.shape))

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
