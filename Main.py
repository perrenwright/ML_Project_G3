#Perren Wright
#Cleveland Heart Dataset
#Going through the dataset

import numpy
import pandas as pd
import sys
import graphviz
from sklearn import tree
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
print("Clevland Heart Disease Dataset")
#reads in values from the url 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#lists out the relvant data set titles 
attributes = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
"oldpeak", "slope", "ca", "thal", "num"]
#reads the url into the data frame
dataframe = pd.read_csv(url, names = attributes)
#prints the dataframe values
#removes non numeric values 
df = dataframe.apply(pd.to_numeric, errors='coerce')
#retains the values from the array
array = df.values
#splits that dataset into inputs and corresponding results
X = array[:303, 0:13]
Y = array[:303, 13]
#transformed to fit the mean value of the column
imputer = Imputer()
X_transf = imputer.fit_transform(X)
#splitting the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_transf, Y, test_size = 0.4)
#building the classifier
clf = tree.DecisionTreeClassifier(max_depth=3)
#training the model
clf = clf.fit(X_train, y_train)
#retrieving the results
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(confusion_matrix)
print(report)
print("The score is: " + str(score))
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("Heart Disease")
