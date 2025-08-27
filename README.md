# Heart Disease Prediction using Machine Learning

This project implements a machine learning pipeline to predict heart disease severity using medical attributes from the HeartDiseaseTrain-Test dataset.

## Dataset Description

The dataset contains 102 records with 13 medical features and 1 target variable:

**Features:**
- `age`: Age of the patient
- `sex`: Gender (1 = male, 0 = female)
- `chest_pain_type`: Type of chest pain (1-4)
- `resting_blood_pressure`: Resting blood pressure
- `cholestoral`: Serum cholesterol in mg/dl
- `fasting_blood_sugar`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `rest_ecg`: Resting electrocardiographic results (0-2)
- `Max_heart_rate`: Maximum heart rate achieved
- `exercise_induced_angina`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (1-3)
- `vessels_colored_by_flourosopy`: Number of major vessels colored by flourosopy (0-3)
- `thalassemia`: Thalassemia type (3,6,7)

**Target:**
- `target`: Heart disease severity (0-3)

## Project Structure

```
.
├── HeartDiseaseTrain-Test.csv      # Original dataset
├── requirements.txt                # Python dependencies
├── heart_disease_eda.py           # Exploratory Data Analysis
├── heart_disease_preprocessing.py # Data preprocessing
├── heart_disease_model.py         # Machine Learning model
└── README.md                      # This file
```

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis
Run the EDA script to understand the dataset:
```bash
python heart_disease_eda.py
```

This will generate:
- Basic dataset information
- Descriptive statistics
- Target variable distribution
- Correlation analysis
- Feature distributions
- Visualizations (saved as PNG files)

### 2. Data Preprocessing
Prepare the data for machine learning:
```bash
python heart_disease_preprocessing.py
```

This script will:
- Handle missing values
- Encode categorical features
- Perform feature engineering
- Scale numerical features
- Split data into train/test sets
- Save processed data to CSV files

### 3. Model Training and Evaluation
Train and evaluate the machine learning model:
```bash
python heart_disease_model.py
```

This will:
- Load processed data
- Train a Random Forest classifier
- Evaluate model performance
- Save the trained model to `heart_disease_model.pkl`

## Machine Learning Approach

### Model Used
- **Algorithm**: Random Forest Classifier
- **Reason**: Handles both numerical and categorical features well, provides feature importance, and is robust to overfitting

### Evaluation Metrics
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, f1-score)

### Expected Output
The model will output:
- Confusion matrix showing predictions vs actual values
- Detailed classification report
- Overall accuracy score

## Results

After running the complete pipeline, you can expect:
1. Visualizations from EDA showing data distributions and correlations
2. Processed datasets ready for machine learning
3. Model performance metrics on the test set
4. A trained model file for future predictions

## Future Enhancements

Potential improvements:
- Try different machine learning algorithms (XGBoost, SVM, Neural Networks)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for more robust evaluation
- Feature selection techniques
- Deployment as a web application

## Dependencies

- pandas==1.5.3
- numpy==1.24.3
- scikit-learn==1.2.2
- matplotlib==3.7.1
- seaborn==0.12.2
- jupyter==1.0.0
- scipy==1.10.1

## License

This project is for educational purposes. Feel free to use and modify as needed.
