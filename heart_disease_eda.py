import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load and return the heart disease dataset"""
    df = pd.read_csv('HeartDiseaseTrain-Test.csv')
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("="*50)
    print("DATASET BASIC INFORMATION")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDataset info:")
    df.info()

def descriptive_stats(df):
    """Display descriptive statistics"""
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)
    print(df.describe())

def target_distribution(df):
    """Analyze target variable distribution"""
    print("\n" + "="*50)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*50)
    target_counts = df['target'].value_counts()
    print(f"Target value counts:\n{target_counts}")
    
    # Calculate percentages separately
    target_percentages = (target_counts / len(df) * 100).round(2)
    print(f"\nTarget value percentages:\n{target_percentages}%")
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='target', data=df)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Heart Disease Severity (0-3)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Correlation with target
    target_corr = corr_matrix['target'].sort_values(ascending=False)
    print(f"Correlation with target:\n{target_corr}")
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_distributions(df):
    """Plot distributions of numerical features"""
    numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 
                         'Max_heart_rate', 'oldpeak']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
    
    # Remove empty subplot
    if len(numerical_features) < 6:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def categorical_analysis(df):
    """Analyze categorical features"""
    categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                           'rest_ecg', 'exercise_induced_angina', 'slope', 
                           'vessels_colored_by_flourosopy', 'thalassemia']
    
    print("\n" + "="*50)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*50)
    
    for feature in categorical_features:
        print(f"\n{feature} value counts:")
        print(df[feature].value_counts())

def outlier_analysis(df):
    """Detect and analyze outliers"""
    numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 
                         'Max_heart_rate', 'oldpeak']
    
    print("\n" + "="*50)
    print("OUTLIER ANALYSIS")
    print("="*50)
    
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        print(f"{feature}: {len(outliers)} outliers detected")

def main():
    """Main function to run EDA"""
    print("Heart Disease Dataset - Exploratory Data Analysis")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Perform analysis
    basic_info(df)
    descriptive_stats(df)
    target_distribution(df)
    correlation_analysis(df)
    categorical_analysis(df)
    outlier_analysis(df)
    feature_distributions(df)
    
    print("\nEDA completed successfully!")
    print("Visualizations saved as PNG files.")

if __name__ == "__main__":
    main()
