import sklearn
from sklearn import datasets
from sklearn import svm
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('adult.data')
data = data['age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,salary'.split(',')]

le = LabelEncoder()
workclass = le.fit_transform(data['workclass'])
education = le.fit_transform(data['education'])
maritalStatus = le.fit_transform(data['marital-status'])
occupation = le.fit_transform(data['occupation'])
relationship = le.fit_transform(data['relationship'])
race = le.fit_transform(data['race'])
sex = le.fit_transform(data['sex'])
nativeCountry = le.fit_transform(data['native-country'])
salary = le.fit_transform(data['salary'])

x = list(zip(data['age'],workclass,data['fnlwgt'],education,data['education-num'],maritalStatus,occupation,relationship,race,sex,data['capital-gain'],data['capital-loss'],data['hours-per-week'],nativeCountry))
y = list(salary)

svm = svm.SVC(C=2)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
salary = ['>50K','<=50K']
svm.fit(x_train,y_train)
predictions = svm.predict(x_test)
for i in range(len(predictions)):
    if salary[y_test[i]] != salary[predictions[i]]:
        correct ='--'
    else:correct=''
    print(f'Actual: {salary[y_test[i]]}        Predicted: {salary[predictions[i]]}  {correct}')
acc = accuracy_score(y_test,predictions)
print(acc)