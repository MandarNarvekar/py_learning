import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
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

# 3D plot: Charges vs BMI vs Age
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = (
    df['smoker']
      .astype(str)
      .str.strip()
      .str.lower()
      .map({'yes': 'red', 'no': 'blue'})
      .fillna('gray')
)
ax.scatter(df['bmi'], df['age'], df['charges'], c=colors, alpha=0.6)
ax.set_xlabel('BMI')
ax.set_ylabel('Age')
ax.set_zlabel('Charges')
ax.set_title('3D Plot: Charges vs BMI and Age (Colored by Smoker Status)')
plt.show()    


# Correlation heatmap
plt.figure(figsize=(10, 8)) 
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')        
plt.show()
# Pairplot to visualize relationships
sns.pairplot(df, hue='smoker', diag_kind='kde', markers=["o", "s"])
plt.suptitle('Pairplot of Insurance Dataset', y=1.02)               
plt.show()  
# Violin plot for age distribution by smoker status
plt.figure(figsize=(10, 6))             
sns.violinplot(data=df, x='smoker', y='age', palette='Set2')
plt.title('Age Distribution by Smoker Status')      
plt.xlabel('Smoker')
plt.ylabel('Age')
plt.show()

