import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

def load_dataset(url):
    return pd.read_csv(url)

def summarize_dataset(iris):
    print(iris.shape)
    print(iris.head(10))
    print(iris.describe())
    print(iris.groupby('class').size())    

def print_plot_univariate(iris):
    print(iris.hist())

def print_plot_multivariate(iris):
    print(scatter_matrix(iris))

def  my_print_and_test_models(iris):
    array = iris.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    cv_results_1 = cross_val_score(DecisionTreeClassifier(), X_train, Y_train,cv = 5, scoring='accuracy')
    cv_results_2 = cross_val_score(GaussianNB(), X_train, Y_train,cv = 5, scoring='accuracy')
    cv_results_3 = cross_val_score(KNeighborsClassifier(), X_train, Y_train,cv = 5, scoring='accuracy')
    cv_results_4 = cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_train, Y_train,cv = 5, scoring='accuracy')
    cv_results_5 = cross_val_score(LinearDiscriminantAnalysis(), X_train, Y_train,cv = 5, scoring='accuracy')
    cv_results_6 = cross_val_score(SVC(gamma='auto'), X_train, Y_train,cv = 5, scoring='accuracy')
    print('%s: %f (%f)' % (\"DecisionTree:\", cv_results_1.mean(), cv_results_1.std()))
    print('%s: %f (%f)' % (\"GaussianNB:\", cv_results_2.mean(), cv_results_2.std()))
    print('%s: %f (%f)' % (\"KNeighbors:\", cv_results_3.mean(), cv_results_3.std()))
    print('%s: %f (%f)' % (\"LogisticRegression:\", cv_results_4.mean(), cv_results_4.std()))
    print('%s: %f (%f)' % (\"LinearDiscriminant:\", cv_results_5.mean(), cv_results_5.std()))
    print('%s: %f (%f)' % (\"SVM:\", cv_results_6.mean(), cv_results_6.std()))

data = load_dataset(\"https://storage.googleapis.com/qwasar-public/track-ds/iris.csv\")