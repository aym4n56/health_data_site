from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

def train_sleep_model():
    # Load and prepare the dataset
    file_path = 'new_health_data.csv'
    data = pd.read_csv(file_path)

    # Drop columns not needed for the prediction
    data = data.drop(columns=['Blood Pressure', 'test', 'High_Blood_Pressure', 'low_Blood_Pressure', 'Stress Level', 'DS_Classification'])
    
    # Drop the first column (if it’s an index or unnamed column)
    data = data.drop(data.columns[0], axis=1)
    
    # Drop rows with missing values
    data = data.dropna()

    # Convert categorical columns to numeric
    label_encoders = {}
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Define features and target
    X_sleep = data.drop(columns=['Quality of Sleep'])
    y_sleep = data['Quality of Sleep']

    # Split the data
    X_train_sleep, X_test_sleep, y_train_sleep, y_test_sleep = train_test_split(X_sleep, y_sleep, test_size=0.2, random_state=42)

    # Train the Quality of Sleep model
    sleep_model = LinearRegression()
    sleep_model.fit(X_train_sleep, y_train_sleep)

    # Evaluate the model
    y_train_sleep_pred = sleep_model.predict(X_train_sleep)
    y_test_sleep_pred = sleep_model.predict(X_test_sleep)

    mse_train_sleep = mean_squared_error(y_train_sleep, y_train_sleep_pred)
    mse_test_sleep = mean_squared_error(y_test_sleep, y_test_sleep_pred)
    r2_train_sleep = r2_score(y_train_sleep, y_train_sleep_pred)
    r2_test_sleep = r2_score(y_test_sleep, y_test_sleep_pred)

    print(f"Training Mean Squared Error (Quality of Sleep): {mse_train_sleep:.2f}")
    print(f"Testing Mean Squared Error (Quality of Sleep): {mse_test_sleep:.2f}")
    print(f"Training R^2 Score (Quality of Sleep): {r2_train_sleep:.2f}")
    print(f"Testing R^2 Score (Quality of Sleep): {r2_test_sleep:.2f}")

    # Get feature names in the order they were trained
    feature_names = X_sleep.columns.tolist()
    
    return sleep_model, label_encoders, feature_names

# Initialize the model and label encoders
sleep_model, label_encoders, feature_names = train_sleep_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract data from the request
    features = [
        data['gender'],
        data['age'],
        data['occupation'],
        data['sleep_duration'],
        data['physical_activity_level'],
        data['bmi_category'],
        data['heart_rate'],
        data['daily_steps'],
        data['sleep_disorder']
    ]
    
    # Ensure the order of features matches the model's expected order
    df = pd.DataFrame([features], columns=feature_names)
    
    # Debug: Print DataFrame columns and their order
    print("DataFrame columns for prediction:", df.columns.tolist())
    
    # Debug: Print the first row of the DataFrame
    print("DataFrame values for prediction:", df.iloc[0].tolist())
    
    try:
        # Predict Quality of Sleep
        quality_of_sleep = sleep_model.predict(df)[0]
        
        # Ensure the quality of sleep is within the range 1-10
        quality_of_sleep = np.clip(quality_of_sleep, 1, 10)
        
        # Send results as JSON
        return jsonify({
            'quality_of_sleep': quality_of_sleep,
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
