import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(".\data.csv")
print(df.head())


# Remove rows with ANY null values
df = df.dropna()

# Remove rows where specific column has null values
#df = df.dropna(subset=['date_column'])

# Remove rows where ALL values are null
#df = df.dropna(how='all')

# Remove columns with null values
#df = df.dropna(axis=1)

# Fill null values instead of removing
#df = df.fillna(0)  # Replace with 0
#df = df.fillna(method='ffill')  # Forward fill
#df = df.fillna(df.mean())  # Replace with column mean
#print(df)


# Find rows where date column doesn't match date format
#df['is_valid_date'] = pd.to_datetime(df['Date'], errors='coerce').notna()

##invalid_dates = df[~df['is_valid_date']]
#print("Invalid dates:\n", invalid_dates)

# Convert Date column to datetime format (replaces the column)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows with invalid dates (NaT values)
df = df.dropna(subset=['Date'])

print("Data with valid dates:\n", df)

# Convert Date column to datetime format (replaces the column)
#df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Remove rows with invalid dates (NaT values)
#df = df.dropna(subset=['Date'])

#print("Data with valid dates:\n", df)

# Optional: Change date format for display
#df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

# Optional: Replace entire column with a specific date
# df['Date'] = pd.to_datetime('2026-01-08')

# Optional: Replace with a date range
# df['Date'] = pd.date_range('2026-01-01', periods=len(df), freq='D')

df = df.drop_duplicates()
# Remove duplicates based on all columns EXCEPT one
# Example: ignore 'Date' column when checking for duplicates
columns_to_check = [col for col in df.columns if col != '']
df = df.drop_duplicates(subset=columns_to_check)


print("Cleaned Data:\n", df)

# Create a simple plot
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Calories'])  # Replace 'column_name' with your column
plt.xlabel('Date')
plt.ylabel('Calories')
plt.title('Calories over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Calories'], bins=20, edgecolor='black')
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.title('Calories Distribution')
plt.tight_layout()
plt.show()

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Calories'], df['Maxpulse'])
plt.xlabel('Calories')
plt.ylabel('Maxpulse')
plt.title('Calories vs Maxpulse')
plt.tight_layout()
plt.show()

# Create a correlation visualization between Calories and Maxpulse
correlation = df[['Calories', 'Maxpulse']].corr()
print("\nCorrelation between Calories and Maxpulse:\n", correlation)

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks([0, 1], ['Calories', 'Maxpulse'])
plt.yticks([0, 1], ['Calories', 'Maxpulse'])
plt.title('Correlation Heatmap: Calories vs Maxpulse')
for i in range(len(correlation)):
    for j in range(len(correlation)):
        plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black', fontsize=12)
plt.tight_layout()
plt.show()




