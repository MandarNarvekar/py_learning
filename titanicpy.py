import pandas as pd

df = pd.read_csv('.\Titanic-Dataset.csv')



#print(df.isnull().sum().sum())

    # Fill missing 'Age' values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Fill missing 'Embarked' values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Convert 'Embarked' column to numerical values
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}) 
        
    # Convert 'Sex' column to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    


print(df.head())
# Display the first few rows of the preprocessed DataFrame
# Preprocess the Titanic dataset
print(df.columns)

#Create a pairplot to visualize relationships between 'Age', 'Fare', and 'Survived'  
import seaborn as sns
import matplotlib.pyplot as plt 
# sns.pairplot(df, vars=['Age', 'Fare'], hue='Survived')      
# plt.show()

#plot correlation heatmap

plt.figure(figsize=(10, 8)) 
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap of Titanic Dataset')
plt.show()
 


