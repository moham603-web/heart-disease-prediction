import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def load_data():
    """Load the processed data"""
    X_train = pd.read_csv('X_train_processed.csv')
    X_test = pd.read_csv('X_test_processed.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    y_test = pd.read_csv('y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the Random Forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_pred = model.predict(X_test)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

def save_model(model, filename='heart_disease_model.pkl'):
    """Save the trained model to a file"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    """Main function to run the model training and evaluation"""
    print("Heart Disease Prediction Model")
    print("="*50)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model)

if __name__ == "__main__":
    main()
