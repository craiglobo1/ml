import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pickle

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target


x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.5)

classes = ['maligant','benign']

clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
# with open('F:\craigComp\Programming\python\ml\smv\bestMod.pickle','rb') as rf:
#     clf = pickle.load(rf)

predictions = clf.predict(x_test)
acc = sklearn.metrics.accuracy_score(y_test,predictions)

for i in range(len(predictions)):
    print(f'Actual:{classes[y_test[i]]}     Predicted:{classes[predictions[i]]}')

print(f'Accuracy:{acc*100}%')
    