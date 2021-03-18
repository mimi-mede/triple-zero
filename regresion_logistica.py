import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Leer datos
df = pd.read_csv('C:\\coffee\\diabetes.csv')
df.head()
columns=['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
X = df[columns].values
y = df['Diabetic'].values

#Normalizar datos
X = preprocessing.StandardScaler().fit(X).transform(X)

#Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

#Entrenar modelo
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
LR.coef_


X.__dict__()


#Predecir
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print (yhat [0:20])
print (y_test [0:20])

#Evaluar
print("Test Accuracy:", accuracy_score(y_test, yhat))
print("Confusion matrix:\n", confusion_matrix(y_test, yhat, labels=[1,0]))
print ("Classification report:\n",classification_report(y_test, yhat))
