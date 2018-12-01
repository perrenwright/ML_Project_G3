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
#replacing the non-existent values 

print("Halberman's")
# Function importing Dataset 
def importdata(): 
    haberman_data = pd.read_csv('https://raw.githubusercontent.com/Kurian-lalan/EDA-on-haberman-survival-dataset/master/haberman.csv') 
    
    
    print ("Dataset Length: ", len(haberman_data)) 
    print ("Dataset Shape: ", haberman_data.shape) 
    
    # Printing the dataset obseravtions 
    print ("Dataset: ",haberman_data.head()) 
    return haberman_data 
 

# Function to split the dataset 
def splitdataset(haberman_data): 
  
    # Seperating the target variable 
    X = haberman_data.values[:, 0:2] 
    Y = haberman_data.values[:, 3]  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.4, random_state = 204) 
      
    return X, Y, X_train, X_test, y_train, y_test
     

# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( criterion = "entropy", random_state = 100, max_depth = 8, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
 

# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
   
# Driver code 
def main(): 
      
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
 
    clf_entropy = train_using_entropy(X_train, X_test, y_train) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
    
    
     

# Calling main function 
if __name__=="__main__": 
    main()

print("Banknote")
authentication = "/home/zizibaby/Documents/Machine Learning/Final Project/banknote1.csv" 
balance_data = pd.read_csv(authentication, sep = ',', header=0)
balance_data.head()
# Function importing Dataset 
def importdata1(): 
    balance_data = pd.read_csv( authentication,

    sep= ',', header =0) 
      
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: " , balance_data.head())
    return balance_data 
  
# Function to split the dataset 
def splitdataset1(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.4, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex. 
def train_using_gini1(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=6, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy. 
def tarin_using_entropy1(X_train, X_test, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 6, min_samples_leaf = 5) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
  
# Function to make predictions 
def prediction1(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy1(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
  
# Driver code 
def main1(): 
      
    # Building Phase 
    data = importdata1() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset1(data) 
    clf_gini = train_using_gini1(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy1(X_train, X_test, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
      
    # Prediction using gini 
    y_pred_gini = prediction1(X_test, clf_gini) 
    cal_accuracy1(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    # Prediction using entropy 
    y_pred_entropy = prediction1(X_test, clf_entropy) 
    cal_accuracy1(y_test, y_pred_entropy) 
      
      
# Calling main function 
if __name__=="__main1__": 
    main1() 