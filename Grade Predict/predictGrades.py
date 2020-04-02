import pandas as pd
import numpy as np
import sklearn 
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import numpy as np
from math import pow,sqrt

def r(x,y):
    xSquare = sum([val*val for val in x])
    ySquare = sum([val*val for val in y])
    SumXY = sum([x[i]*y[i] for i in range(len(x))])
    Sxx = xSquare - (pow(sum(x),2)/len(x))
    Syy = ySquare - (pow(sum(y),2)/len(x))
    Sxy = SumXY - ((sum(x)*sum(y))/len(x))
    r = Sxy/sqrt((Sxx*Syy))
    return r



data = pd.read_csv("student-mat.csv",sep=';',header=0)
attrOG = 'age;traveltime;studytime;failures;schoolsup;higher;goout;G1;G2;G3'
attrOG = attrOG.split(';')
data = data[attrOG]


boolAttrs = 'schoolsup;higher;'.split()
data['schoolsup'] = data['schoolsup'].map({'yes':1,'no':0})
data['higher'] = data['higher'].map({'yes':1,'no':0})


predict = 'G3'

attr = np.array(data.drop([predict],1))
labels = np.array(data[predict])



sumAcc = 0
for i in range(1000):
    attr_train,attr_test,labels_train,labels_test = sklearn.model_selection.train_test_split(attr,labels,test_size=0.15)
    linear = linear_model.LinearRegression()


    linear.fit(attr_train,labels_train)
    acc= linear.score(attr_test,labels_test)
    sumAcc+= acc
    print(acc)
print(f'Average acc:{sumAcc/1000}')










# G1 = np.array(data['G1'])
# G2 = np.array(data['G2'])


# for i in range(len(attrOG)):
#     TempA = np.array(data[attrOG[i]])
#     rCoef1 = r(G1,TempA)
#     rCoef2 = r(G2,TempA) 
#     rCoef3 = r(labels,TempA)
#     if abs(rCoef1) > 0.15 or abs(rCoef2) > 0.15 or abs(rCoef3) > 0.15:
#         print(f'{attrOG[i]} - {rCoef1} {rCoef2} {rCoef3}')











# data = data[['G1','G2','G3','failures','studytime','absences']]

# predict = 'G3'

# attr = np.array(data.drop([predict],1))
# labels = np.array(data[predict])
# attr_train,attr_test,labels_train,labels_test = sklearn.model_selection.train_test_split(attr,labels,test_size=0.15)

# with open('bestModel.pickle','rb') as rf:
#     linear = pickle.load(rf)

# predictions = linear.predict(attr_test)

# for i in range(len(predictions)):
#     print(f'P G3: {predictions[i]}  A G3: {labels_test[i]}  diff:{abs(predictions[i]-labels_test[i])}')

# print(f'm: {linear.coef_}')
# print(f'c: {linear.intercept_}')