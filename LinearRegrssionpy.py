"""
Linear Regression Sample Code
This file demonstrates different approaches to linear regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# Example 1: Simple Linear Regression with scikit-learn
# ==============================================================================
print("=" * 60)
print("Example 1: Simple Linear Regression")
print("=" * 60)

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Feature (must be 2D)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 11])  # Target variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print results
print(f"Coefficient (slope): {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# Example 2: Multiple Linear Regression
# ==============================================================================
print("\n" + "=" * 60)
print("Example 2: Multiple Linear Regression")
print("=" * 60)

# Create dataset with multiple features
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Previous_Score': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Sleep_Hours': [5, 6, 7, 8, 6, 7, 8, 7, 8, 9],
    'Final_Score': [55, 60, 68, 75, 78, 82, 88, 90, 95, 98]
}

df = pd.DataFrame(data)
print("\nDataset:")
print(df)

# Prepare features and target
X_multi = df[['Hours_Studied', 'Previous_Score', 'Sleep_Hours']]
y_multi = df['Final_Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train the model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Make predictions
y_train_pred = model_multi.predict(X_train)
y_test_pred = model_multi.predict(X_test)

# Evaluate the model
print(f"\nCoefficients:")
for feature, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercept: {model_multi.intercept_:.4f}")

print(f"\nTraining R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Testing R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"Testing RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

# ==============================================================================
# Example 3: Manual Implementation of Linear Regression
# ==============================================================================
print("\n" + "=" * 60)
print("Example 3: Manual Linear Regression (Understanding the Math)")
print("=" * 60)

# Simple dataset
X_manual = np.array([1, 2, 3, 4, 5])
y_manual = np.array([2, 4, 5, 4, 5])

# Calculate coefficients using the Normal Equation
# Formula: β = (X^T * X)^-1 * X^T * y
# For simple linear regression: slope = Σ((x - x_mean)(y - y_mean)) / Σ((x - x_mean)²)

x_mean = np.mean(X_manual)
y_mean = np.mean(y_manual)

# Calculate slope (m)
numerator = np.sum((X_manual - x_mean) * (y_manual - y_mean))
denominator = np.sum((X_manual - x_mean) ** 2)
slope = numerator / denominator

# Calculate intercept (b)
intercept = y_mean - slope * x_mean

# Make predictions
y_pred_manual = slope * X_manual + intercept

print(f"Manual Calculation:")
print(f"Slope (m): {slope:.4f}")
print(f"Intercept (b): {intercept:.4f}")
print(f"Equation: y = {slope:.4f}x + {intercept:.4f}")

# Calculate R² manually
ss_total = np.sum((y_manual - y_mean) ** 2)
ss_residual = np.sum((y_manual - y_pred_manual) ** 2)
r2_manual = 1 - (ss_residual / ss_total)
print(f"R² Score: {r2_manual:.4f}")

# ==============================================================================
# Example 4: Real-world Example with Polynomial Features
# ==============================================================================
print("\n" + "=" * 60)
print("Example 4: Polynomial Regression")
print("=" * 60)

from sklearn.preprocessing import PolynomialFeatures

# Generate non-linear data
X_poly = np.linspace(0, 10, 50).reshape(-1, 1)
y_poly = 2 * X_poly**2 + 3 * X_poly + np.random.randn(50, 1) * 10

# Linear regression on original data
model_linear = LinearRegression()
model_linear.fit(X_poly, y_poly)
y_pred_linear = model_linear.predict(X_poly)

# Polynomial regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly_transformed = poly_features.fit_transform(X_poly)
model_poly = LinearRegression()
model_poly.fit(X_poly_transformed, y_poly)
y_pred_poly = model_poly.predict(X_poly_transformed)

print(f"Linear Model R²: {r2_score(y_poly, y_pred_linear):.4f}")
print(f"Polynomial Model R²: {r2_score(y_poly, y_pred_poly):.4f}")

# Visualize comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_poly, y_poly, alpha=0.5, label='Data')
plt.plot(X_poly, y_pred_linear, 'r-', linewidth=2, label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X_poly, y_poly, alpha=0.5, label='Data')
plt.plot(X_poly, y_pred_poly, 'g-', linewidth=2, label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Key Concepts:")
print("=" * 60)
print("""
1. Linear Regression finds the best-fitting straight line through data points
2. Equation: y = mx + b (simple) or y = β₀ + β₁x₁ + β₂x₂ + ... (multiple)
3. R² Score: Measures how well the model fits (0 to 1, higher is better)
4. MSE/RMSE: Measures prediction error (lower is better)
5. Train-Test Split: Essential for evaluating model performance on unseen data
6. Polynomial Regression: Uses linear regression on polynomial features for non-linear data
""")