import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'new_health_data.csv'
data = pd.read_csv(file_path)

# Drop the unnamed column and columns not needed for the prediction
data = data.drop(data.columns[0], axis=1)
data = data.drop(columns=['Blood Pressure', 'test', 'High_Blood_Pressure', 'low_Blood_Pressure', 'Stress Level', 'DS_Classification'])

# Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['BMI Category'] = data['BMI Category'].astype('category').cat.codes
data['Sleep Disorder'] = data['Sleep Disorder'].astype('category').cat.codes

# Define features and target
X = data.drop(columns=['Quality of Sleep'])
y = data['Quality of Sleep']

# Feature scaling (optional but recommended)
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Training Mean Squared Error: {mse_train:.2f}")
print(f"Testing Mean Squared Error: {mse_test:.2f}")
print(f"Training R^2 Score: {r2_train:.2f}")
print(f"Testing R^2 Score: {r2_test:.2f}")

# Plotting feature importance
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X, columns=X.columns).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix (Quality of Sleep Prediction)')
plt.show()

# Save the model and scaler
joblib.dump(model, 'sleep_quality_model.joblib')
joblib.dump(scaler, 'scaler_sleep.joblib')
print(f'Model saved to sleep_quality_model.joblib')
print(f'Scaler saved to scaler_sleep.joblib')
