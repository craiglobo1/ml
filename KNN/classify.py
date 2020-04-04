import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

data = pd.read_csv('F:\craigComp\Programming\python\ml\KNN\car.data')

print('This is a Machine Learning KNearestNeighbors Algorithm which Predicts class of the car\n using some of the data below')
print(f'\n{data.head()}\n')

labelEncode = preprocessing.LabelEncoder()
buying = labelEncode.fit_transform(list(data['buying']))
maint = labelEncode.fit_transform(list(data['maint']))
door = labelEncode.fit_transform(list(data['door']))
persons = labelEncode.fit_transform(list(data['persons']))
lug_boot = labelEncode.fit_transform(list(data['lug_boot']))
safety = labelEncode.fit_transform(list(data['safety']))
cls = labelEncode.fit_transform(list(data['class']))

predict = 'class'

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1) 

k = 7

model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_train,y_train)
print(f'score:{model.score(x_test,y_test)*100}%')

predictions = model.predict(x_test)

names = ['unacc', 'acc', 'good', 'vgood']

for i in range(len(predictions)):
    print(f'actual:{names[y_test[i]]}     predicted:{names[predictions[i]]}        p data:{x_test[i]}')

print(f'score:{model.score(x_test,y_test)*100}%')