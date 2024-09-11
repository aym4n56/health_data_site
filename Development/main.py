import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = 'new_health_data.csv'
data = pd.read_csv(file_path)

# Drop the 'test' column
data = data.drop(columns=['test'])

# Display the first few rows of the dataset
print(data.head())

# Display basic information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Function to standardize categorical values
def standardize_categories(df, column_name, standard_values):
    df[column_name] = df[column_name].str.strip().str.title()  # Standardize casing and strip whitespace
    df[column_name] = df[column_name].replace(standard_values)

# Standardize 'DS_Classification' and other columns
standardize_categories(data, 'DS_Classification', {'Level 1': 'Level 1', 'Level 2': 'Level 2', 'Level 3': 'Level 3'})
standardize_categories(data, 'Blood Pressure', {'Low': 'Low', 'High': 'High'})  # If applicable

# Convert categorical columns to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['BMI Category'] = data['BMI Category'].astype('category').cat.codes
data['Sleep Disorder'] = data['Sleep Disorder'].astype('category').cat.codes
data['DS_Classification'] = data['DS_Classification'].astype('category').cat.codes

# Split 'Blood Pressure' into two separate columns: 'Systolic' and 'Diastolic'
data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True)
data['Systolic'] = pd.to_numeric(data['Systolic'])
data['Diastolic'] = pd.to_numeric(data['Diastolic'])

# Drop the original 'Blood Pressure' column
data = data.drop(columns=['Blood Pressure'])

# Check the updated data types
print(data.dtypes)

# Drop rows with missing values as an example
data = data.dropna()

# Define features and target
X = data.drop(columns=['High_Blood_Pressure'])
y = data['High_Blood_Pressure']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
