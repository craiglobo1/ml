import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from math import sqrt
import pickle

def r(x,y):
    xSquare = sum([val*val for val in x])
    ySquare = sum([val*val for val in y])
    SumXY = sum([x[i]*y[i] for i in range(len(x))])
    Sxx = xSquare - (pow(sum(x),2)/len(x))
    Syy = ySquare - (pow(sum(y),2)/len(x))
    Sxy = SumXY - ((sum(x)*sum(y))/len(x))
    #  Sxx*Syy = -ve num and sqrt of -ve no is imaginary
    r = Sxy/sqrt(Sxx*Syy)
    return r

data = pd.read_csv('F:\craigComp\Programming\python\ml\winequality-red.csv',sep=';')
heading = 'fixed acidity;citric acid;residual sugar;chlorides;density;pH;sulphates;alcohol;quality'.split(';')
# data = data[heading]


predict = 'citric acid'

x = np.array(data.drop([predict],1))
y = np.array(data[predict])


# for i in range(9):
#     TempA = np.array(data[heading[i]]) 
#     rCoeff = r(TempA,y)
#     if rCoeff > 0.1:
#         print(f'{heading[i]} - {rCoeff}')

best = 0
for i in range(1000):

    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.15)

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc > best:
        acc = best
        with open('BestWine.pickle','wb+') as wf:
            pickle.dump(linear,wf)

with open('BestWine.pickle','rb+') as rf:
    linear = pickle.load(rf)
acc = linear.score(x_test,y_test)
print(acc)
