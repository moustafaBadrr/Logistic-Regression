#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#upload dataset file
from google.colab import files
#datasetFile = files.upload()

#loading the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")

# divide the data into x and y
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values 

#take training and test sets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = 0.3, random_state = 0)

#scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xTest = sc_x.fit_transform(xTest)
xTrain = sc_x.fit_transform(xTrain)

#fit the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xTrain,yTrain)

#prediction
yPrediction = model.predict(xTest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(yTest, yPrediction)#two types of people 1 - buy 2 - not buy

