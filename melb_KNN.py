import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score



veriler = pd.read_csv('melb_data.csv')
corr=veriler.corr()
describe=veriler.describe()

print(veriler.isnull().sum())
veriler=veriler.dropna()

girisler=['Rooms','Distance','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']
x=veriler[girisler]
y=veriler[['Price']]

x_train, x_test, y_train, y_test=train_test_split(x, y,test_size=0.10)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

knn= KNeighborsClassifier(n_neighbors=2, metric='euclidean', weights='distance')
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)


r2=r2_score(y_test, y_pred)
print('R2 score:')
print(r2)