import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'new_health_data.csv'
data = pd.read_csv(file_path)

# Drop the unnamed column and columns not used in the prediction
data = data.drop(data.columns[0], axis=1)
data = data.drop(columns=['Blood Pressure', 'test', 'High_Blood_Pressure', 'low_Blood_Pressure', 'DS_Classification'])

# Drop rows with missing values
data = data.dropna()

# Convert categorical columns to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['BMI Category'] = data['BMI Category'].astype('category').cat.codes
data['Sleep Disorder'] = data['Sleep Disorder'].astype('category').cat.codes

# Define features and target
X = data.drop(columns=['Stress Level'])
y = data['Stress Level']

# Feature scaling (optional but recommended)
scaler = MinMaxScaler(feature_range=(0, 10))
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print evaluation metrics
print(f'Training Mean Squared Error: {train_mse:.2f}')
print(f'Testing Mean Squared Error: {test_mse:.2f}')
print(f'Training R^2 Score: {train_r2:.2f}')
print(f'Testing R^2 Score: {test_r2:.2f}')

# Save the model and scaler
joblib.dump(model, 'stress_level_model.joblib')
joblib.dump(scaler, 'scaler_stress.joblib')
print(f'Model saved to stress_level_model.joblib')
print(f'Scaler saved to scaler_stress.joblib')
