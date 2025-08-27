import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the heart disease dataset"""
    df = pd.read_csv('HeartDiseaseTrain-Test.csv')
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Checking for missing values...")
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values[missing_values > 0]}")
    
    # If there are missing values, impute them
    if missing_values.sum() > 0:
        # For numerical columns, use mean imputation
        numerical_cols = ['age', 'resting_blood_pressure', 'cholestoral', 
                         'Max_heart_rate', 'oldpeak']
        
        # For categorical columns, use mode imputation
        categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                           'rest_ecg', 'exercise_induced_angina', 'slope', 
                           'vessels_colored_by_flourosopy', 'thalassemia']
        
        # Create imputers
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Impute numerical columns
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        
        # Impute categorical columns
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        
        print("Missing values imputed successfully!")
    
    return df

def encode_categorical_features(df):
    """Encode categorical features if needed"""
    # Most features appear to be already encoded numerically
    # Check if any categorical features need encoding
    categorical_cols = ['chest_pain_type', 'rest_ecg', 'slope', 
                       'vessels_colored_by_flourosopy', 'thalassemia']
    
    print("Categorical features encoding status:")
    for col in categorical_cols:
        unique_values = df[col].unique()
        print(f"{col}: {len(unique_values)} unique values - {sorted(unique_values)}")
    
    # The categorical features appear to be already numerically encoded
    # No additional encoding needed based on the data exploration
    
    return df

def feature_engineering(df):
    """Create new features if beneficial"""
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 40, 50, 60, 70, 100], 
                            labels=['<40', '40-50', '50-60', '60-70', '70+'])
    
    # Create blood pressure categories
    df['bp_category'] = pd.cut(df['resting_blood_pressure'],
                              bins=[0, 120, 140, 160, 200, 300],
                              labels=['Normal', 'Elevated', 'High1', 'High2', 'Crisis'])
    
    # Create cholesterol categories
    df['chol_category'] = pd.cut(df['cholestoral'],
                                bins=[0, 200, 240, 300, 400, 600],
                                labels=['Normal', 'Borderline', 'High', 'Very High', 'Extreme'])
    
    # Convert categorical features to dummy variables
    df = pd.get_dummies(df, columns=['age_group', 'bp_category', 'chol_category'], 
                       drop_first=True)
    
    return df

def scale_features(df, features_to_scale):
    """Scale numerical features"""
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df, scaler

def prepare_data_for_ml(df, test_size=0.2, random_state=42):
    """Prepare data for machine learning"""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target distribution in training set:\n{y_train.value_counts()}")
    print(f"Target distribution in test set:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main preprocessing function"""
    print("Heart Disease Dataset - Data Preprocessing")
    print("="*50)
    
    # Load data
    df = load_data()
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Feature engineering
    df = feature_engineering(df)
    print(f"Dataset shape after feature engineering: {df.shape}")
    
    # Identify numerical features to scale
    numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 
                         'Max_heart_rate', 'oldpeak']
    
    # Scale features
    df, scaler = scale_features(df, numerical_features)
    print("Features scaled successfully!")
    
    # Prepare data for ML
    X_train, X_test, y_train, y_test = prepare_data_for_ml(df)
    
    # Save processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }
    
    # Save to files (optional - for larger datasets)
    X_train.to_csv('X_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("\nData preprocessing completed successfully!")
    print("Processed data saved to CSV files.")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()
