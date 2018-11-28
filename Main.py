#Perren Wright
#Cleveland Heart Dataset
#Going through the dataset
import pandas
import numpy
import sys
print("Clevland Heart Disease Dataset")
#reads in values from the url 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#lists out the relvant data set titles 
attributes = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
"oldpeak", "slope", "ca", "thal", "num"]
#reads the url into the data frame
dataframe = pandas.read_csv(url, names = attributes)
#prints the dataframe values 
print (dataframe)
#replacing the non-existent values 
print("banknote authentication Data Set")
print("habermans survival")
