import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'new_health_data.csv'
data = pd.read_csv(file_path)

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

# Convert categorical columns to numeric
data['Gender'] = data['Gender'].astype('category').cat.codes
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['BMI Category'] = data['BMI Category'].astype('category').cat.codes
data['Sleep Disorder'] = data['Sleep Disorder'].astype('category').cat.codes
data['DS_Classification'] = data['DS_Classification'].astype('category').cat.codes

# Drop the 'Blood Pressure' column and any other unwanted columns
data = data.drop(columns=['Blood Pressure', 'test', 'High_Blood_Pressure', 'low_Blood_Pressure'])

# Check the updated data types
print(data.dtypes)

# Drop rows with missing values as an example
data = data.dropna()

# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=[float, int])

# Compute the correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

def evaluate_feature_predictability(target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Evaluate predictability for each feature
results = {}
for column in numeric_data.columns:
    if column not in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder', 'DS_Classification']:  # Exclude categorical features
        mse = evaluate_feature_predictability(column)
        results[column] = mse

# Print results
sorted_results = sorted(results.items(), key=lambda x: x[1])
print("Features sorted by Mean Squared Error (lower is better):")
for feature, mse in sorted_results:
    print(f"{feature}: {mse}")

# Identify the most predictable feature
most_predictable_feature = sorted_results[0][0]  # Feature with the lowest MSE
print(f"The most predictable feature is: {most_predictable_feature}")

# Example: Visualize the relationship between the most predictable feature and one of the predictors
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y=most_predictable_feature)
plt.title(f'Relationship between Age and {most_predictable_feature}')
plt.xlabel('Age')
plt.ylabel(most_predictable_feature)
plt.show()
