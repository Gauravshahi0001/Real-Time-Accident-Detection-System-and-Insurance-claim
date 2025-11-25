import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Sample data generation (replace with real data when available)
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    
    vehicle_count = np.random.randint(1, 5, n_samples)
    injuries = np.random.randint(0, 10, n_samples)
    time_of_day = np.random.randint(0, 24, n_samples)
    weather_condition = np.random.randint(0, 5, n_samples)  # 0:Clear, 1:Rain, 2:Snow, etc.
    
    # Create severity based on features
    severity = np.zeros(n_samples)
    for i in range(n_samples):
        base_severity = 0
        # More vehicles increase severity
        base_severity += vehicle_count[i] * 0.5
        # More injuries increase severity
        base_severity += injuries[i] * 0.3
        # Night time (between 22-5) increases severity
        if time_of_day[i] >= 22 or time_of_day[i] <= 5:
            base_severity += 1
        # Bad weather increases severity
        if weather_condition[i] > 0:
            base_severity += weather_condition[i] * 0.5
            
        # Add some randomness
        base_severity += np.random.normal(0, 0.5)
        severity[i] = np.clip(round(base_severity), 1, 5)
    
    X = np.column_stack([vehicle_count, injuries, time_of_day, weather_condition])
    y = severity
    
    return X, y

def train_model():
    # Generate or load your training data
    X, y = generate_sample_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.2f}")
    print(f"Testing accuracy: {test_score:.2f}")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    joblib.dump(model, 'model/accident_classifier.joblib')
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model()
