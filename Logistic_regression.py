import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load dataset
data = pd.read_csv('insurance.csv')
# Preprocess the data
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables to dummy variables
# Define features and target variable       
X = data.drop('charges', axis=1)  # Features
y = (data['charges'] > data['charges'].median()).astype(int)  # Binary target: 1 if charges > median, else 0
# Split dataset into training and testing sets      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train) 
# Make predictions
y_pred = model.predict(X_test)          
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)       
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print(y_test.head())
print(X_test[0:5])
print('Classification Report:')
print(class_report)