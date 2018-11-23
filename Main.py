#Perren Wright
#Cleveland Heart Dataset
#Going through the dataset
import pandas
import numpy
import sys
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
attributes = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
dataframe = pandas.read_csv(url, names = attributes)
array = dataframe.values
print (array) 
print("Clevland Heart Disease Dataset")

print("banknote authentication Data Set")
print("habermans survival")
