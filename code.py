# Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style for aesthetics
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

print("--- House Price Prediction Notebook ---")
print("Step 1: Setting up the environment and loading data.")

# --- Load the dataset ---
# Make sure you have uploaded 'BostonHousing.csv' to your Colab session.
try:
    df = pd.read_csv('BostonHousing.csv')
    print("\nDataset 'BostonHousing.csv' loaded successfully!")
    print("Dataset shape:", df.shape)
except FileNotFoundError:
    print("\nError: 'BostonHousing.csv' not found.")
    print("Please upload the file to the Colab session by clicking the folder icon on the left.")

# Display the first 5 rows to see the data and column names
print("\nFirst 5 rows of the dataset:")
print(df.head())



# Data Preprocessing and Exploratory Data Analysis (EDA)
print("\n--- Step 2: Data Preprocessing and EDA ---")

# 1. Inspect data types and missing values
print("\nDataset Info:")
df.info()
# The .info() output shows no missing values, so we can proceed.
# All columns are numerical, which simplifies preprocessing.

# 2. Rename columns for clarity (optional but good practice)
# 'rm' is average rooms, 'lstat' is % lower status, 'medv' is median value (our target)
df.rename(columns={'medv': 'price'}, inplace=True)
print("\nRenamed target column 'medv' to 'price'.")

# 3. EDA: Visualize the distribution of the target variable ('price')
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of House Prices (in $1000s)')
plt.xlabel('Price ($1000s)')
plt.ylabel('Frequency')
plt.show()

# 4. EDA: Correlation Heatmap
# See which features are most correlated with the price.
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Boston Housing Features')
plt.show()
print("\nObservation: 'rm' (number of rooms) has a strong positive correlation with price,")
print("while 'lstat' (% lower status of the population) has a strong negative correlation.")



# Model Preparation (Splitting and Scaling)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n--- Step 3: Preparing Data for Modeling ---")

# 1. Define Features (X) and Target (y)
# We will use all features to predict the price.
X = df.drop('price', axis=1)
y = df['price']

# 2. Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set has {X_train.shape[0]} samples.")
print(f"Testing set has {X_test.shape[0]} samples.")

# 3. Scale the features
# Feature scaling is important for distance-based algorithms and linear models.
# We fit the scaler ONLY on the training data to avoid data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures have been successfully scaled.")


# Train Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

print("\n--- Step 4: Training Regression Models ---")

# --- Model 1: Linear Regression ---
print("Training Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("Linear Regression training complete.")

# --- Model 2: Gradient Boosting Regressor ---
# This is a more complex model that often yields better performance.
print("\nTraining Gradient Boosting model...")
gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train_scaled, y_train)
print("Gradient Boosting training complete.")


# Model Evaluation and Visualization
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("\n--- Step 5: Evaluating Model Performance ---")

# --- Make predictions on the test set ---
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_gbr = gbr_model.predict(X_test_scaled)

# --- Evaluate Linear Regression ---
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("\n--- Linear Regression Performance ---")
print(f"Mean Absolute Error (MAE): {mae_lr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lr:.2f}")

# --- Evaluate Gradient Boosting ---
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
print("\n--- Gradient Boosting Performance ---")
print(f"Mean Absolute Error (MAE): {mae_gbr:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_gbr:.2f}")

print("\nNote: MAE represents the average error in price prediction (in $1000s).")
print("RMSE gives more weight to larger errors.")

# --- Visualize Predicted vs. Actual Prices ---
# A perfect model would have all points on the red dashed line.
plt.figure(figsize=(16, 7))

# Plot for Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6, edgecolors='w', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2, label='Perfect Fit')
plt.xlabel("Actual Prices ($1000s)")
plt.ylabel("Predicted Prices ($1000s)")
plt.title("Linear Regression: Actual vs. Predicted")
plt.legend()

# Plot for Gradient Boosting
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_gbr, alpha=0.6, edgecolors='w', color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2, label='Perfect Fit')
plt.xlabel("Actual Prices ($1000s)")
plt.ylabel("Predicted Prices ($1000s)")
plt.title("Gradient Boosting: Actual vs. Predicted")
plt.legend()

plt.tight_layout()
plt.show()

print("\n END OF PROJECT")
