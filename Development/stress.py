import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'new_health_data.csv'
data = pd.read_csv(file_path)

# Drop columns not needed for this prediction
data = data.drop(columns=['Blood Pressure', 'test', 'High_Blood_Pressure', 'low_Blood_Pressure', 'Quality of Sleep'])

# Convert categorical columns to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['BMI Category'] = data['BMI Category'].astype('category').cat.codes
data['Sleep Disorder'] = data['Sleep Disorder'].astype('category').cat.codes
data['DS_Classification'] = data['DS_Classification'].astype('category').cat.codes

# Drop rows with missing values as an example
data = data.dropna()

# Define features and target
X = data.drop(columns=['Stress Level'])
y = data['Stress Level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for predicting Stress Level: {mse:.2f}")

# Plotting feature importance
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix (Stress Level Prediction)')
plt.show()

#mse for predicting stress level: 0.32
