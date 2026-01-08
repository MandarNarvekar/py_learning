import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt 
# Load an example dataset
df = pd.read_csv('insurance.csv')
print(df.head())

df=df.dropna()
# Basic statistics
#print(df.describe())

# Visualize BMI vs insurance charges
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', alpha=0.6)
plt.title('BMI vs Insurance Charges')
plt.xlabel('BMI')
plt.ylabel('Insurance Charges')
plt.show()

# Visualize insurance claims with charges
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='smoker', y='charges')
plt.title('Insurance Charges by Smoker Status')
plt.xlabel('Smoker')
plt.ylabel('Insurance Charges')
plt.show()

# Histogram of charges with age and hue for smokers
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='smoker', multiple='stack', bins=30)
plt.title('Age Distribution by Smoker Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Additional visualization: Charges distribution by smoker status and children
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='charges', hue='smoker', multiple='dodge', bins=30)
plt.title(f'Charges Distribution by Smoker Status (Total Children: {df["children"].sum()})')
plt.xlabel('Charges')
plt.ylabel('Count')
plt.show()    

