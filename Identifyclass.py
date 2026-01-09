import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report     
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load dataset

data= pd.read_csv('colorlabel.csv')  

# Preprocess the data
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables to dummy variables

print(data.head())
# Define features and target variable   
X = data.drop('fruit', axis=1)  # Features
y = data['fruit']  # Target variable

model=LogisticRegression()
model.fit(X,y)
# Make predictions  
y_pred = model.predict(X)
# Evaluate the model
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)
class_report = classification_report(y, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Split dataset into training and testing sets
#   We will use 80% of the data for training and 20% for testing    
#  Setting random_state for reproducibility
# Stratify to maintain class distribution
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Feature scaling
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)   
# Create and train the Logistic Regression model
#model = LogisticRegression()
#model.fit(X_train, y_train)
# Make predictions
#y_pred = model.predict(X_test)
# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)
#conf_matrix = confusion_matrix(y_test, y_pred)
#class_report = classification_report(y_test, y_pred)
#print(f'Accuracy: {accuracy}')
#print('Confusion Matrix:')

