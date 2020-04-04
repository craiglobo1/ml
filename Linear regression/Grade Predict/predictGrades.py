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

print('This is a Machine Learning linear regression Algorithm which Predicts the grade of students\n using some of the data below')

def r(x,y):
    xSquare = sum([val*val for val in x])
    ySquare = sum([val*val for val in y])
    SumXY = sum([x[i]*y[i] for i in range(len(x))])
    Sxx = xSquare - (pow(sum(x),2)/len(x))
    Syy = ySquare - (pow(sum(y),2)/len(x))
    Sxy = SumXY - ((sum(x)*sum(y))/len(x))
    r = Sxy/sqrt((Sxx*Syy))
    return r



data = pd.read_csv("F:\craigComp\Programming\python\ml\Grade Predict\student-mat.csv",sep=';',header=0)
attrOG = 'age;traveltime;studytime;failures;schoolsup;higher;goout;G1;G2;G3'
attrOG = attrOG.split(';')
data = data[attrOG]
print(data.head())

boolAttrs = 'schoolsup;higher;'.split()
data['schoolsup'] = data['schoolsup'].map({'yes':1,'no':0})
data['higher'] = data['higher'].map({'yes':1,'no':0})


predict = 'G3'

attr = np.array(data.drop([predict],1))
labels = np.array(data[predict])
attr_train,attr_test,labels_train,labels_test = sklearn.model_selection.train_test_split(attr,labels,test_size=0.15)

# best = 0
# sumAcc = 0
# for i in range(1000):
#     attr_train,attr_test,labels_train,labels_test = sklearn.model_selection.train_test_split(attr,labels,test_size=0.15)
#     linear = linear_model.LinearRegression()


#     linear.fit(attr_train,labels_train)
#     acc= linear.score(attr_test,labels_test)
#     sumAcc+= acc
#     print(acc)
#     if acc > best:
#         acc = best
#         with open(r'F:\craigComp\Programming\python\ml\Grade Predict\bestModel.pickle','wb') as wf:
#             pickle.dump(linear,wf)

with open(r'F:\craigComp\Programming\python\ml\Grade Predict\bestModel.pickle','rb') as rf:
    linear = pickle.load(rf)

predictions = linear.predict(attr_test)

for i in range(len(predictions)):
    print(f'Actual G3:{labels_test[i]}         Predicted G3:{round(predictions[i])}')

# print(f'Average acc:{sumAcc/1000}')
print(f'Accuracy: {linear.score(attr_train,labels_train)*100}%')

print(f'm: {linear.coef_}')
print(f'c: {linear.intercept_}')

# x = np.linspace(-5,20,100)
# y = (linear.coef_[7])*x+(linear.intercept_)
# plt.plot(x, y, '-r', label='y=2x+1')

# p = 'G1'
# style.use('ggplot')
# plt.scatter(data[p],data['G3'])
# plt.xlabel(p)
# plt.ylabel('G3 / Marks')
# plt.show()