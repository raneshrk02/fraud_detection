# Fraud Detection System

## Project Overview
This project implements a machine learning system to detect fraudulent credit card transactions. The system processes transaction data, extracts meaningful features, and trains multiple models to identify patterns indicative of fraudulent activity while minimizing false positives.

## Directory Structure
```
fraud_detection/
|
|--- datasets/                    # Contains all data files
|   |-- fraudTrain.csv            # Original training dataset
|   |-- fraudTest.csv             # Original test dataset
|   |-- train_df.csv              # Processed training data
|   |-- test_df.csv               # Processed test data
|
|--- utils/                       # Utility scripts and notebooks
|   |-- model.ipynb               # Model training and evaluation
|   |-- EDA.ipynb                 # Exploratory data analysis
|
|--- requirements.txt             # Project dependencies
|---README.md                     # Project documentation
```
Note: './datasets' would be empty before running the code, dataset is stored in this directory once it is downloaded/created.

## Dataset Description
The dataset is downloaded from kaggle and contains credit card transaction information with the following features:
- Transaction details (date, time, amount)
- Customer information (name, gender, location, etc.)
- Merchant information (name, category, location)
- Binary target variable indicating fraud (is_fraud)

Dataset statistics:
- Training set: 1,296,675 transactions
- Test set: 555,719 transactions
- Class imbalance: Fraudulent transactions represent a small percentage of the total (0.58%)

## Methodology

### 1. Data Exploration & Preprocessing
- Analyzed distribution of transaction amounts, times, and locations
- Explored relationships between features and fraud status
- Visualized geographical patterns of transactions
- Handled missing values and outliers

### 2. Feature Engineering
- Created time-based features (hour, day, weekend flag, etc.)
- Calculated geographical distance between customer and merchant
- Generated customer profiling features (average transaction amount)
- Derived merchant and category risk scores
- Engineered transaction recency and relative amount features

### 3. Model Development
- Implemented preprocessing pipeline with proper scaling and encoding
- Applied SMOTE to address class imbalance
- Trained multiple models:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
- Performed cross-validation and hyperparameter optimization
- Evaluated models using precision-recall AUC and ROC AUC

### 4. Threshold Optimization
- Implemented cost-sensitive threshold optimization
- Balanced false positive and false negative costs
- Fine-tuned threshold to minimize overall business impact

### 5. Misclassification Analysis
- Identified patterns in false positives and false negatives
- Analyzed feature importance for misclassified transactions
- Provided insights for further model improvements

## Results
- Achieved high fraud detection rate with manageable false positive rate
- Identified key indicators of fraudulent behavior:
  - Transaction distance
  - Unusual transaction amounts
  - Merchant risk profiles
  - Time patterns
- Developed robust scoring system for real-time transaction evaluation

## How to Use

### Prerequisites
Install required dependencies:
```
pip install -r requirements.txt
```

### Exploratory Data Analysis
Run the EDA notebook to explore the dataset:
```
jupyter notebook utils/EDA.ipynb
```

### Model Training and Evaluation
Execute the modeling notebook:
```
jupyter notebook utils/model.ipynb
```

## Key Insights
1. Transportation and gas-related transactions show significantly higher risk of fraud
2. Transaction amount is a strong predictor, with unusual amounts relative to customer behavior flagged
3. Time-based patterns are important, with evening hours showing elevated fraud rates
4. Category-based risk profiles provide valuable context for transaction evaluation
5. Fraudulent transactions often show distinct patterns in:
   - Transaction timing (certain hours show higher fraud rates)
   - Geographic patterns (greater distances between customer and merchant)
   - Specific merchant categories

6. Model performance varies by transaction type and amount:
   - Higher detection rates for larger transaction amounts
   - Category-specific patterns require specialized detection rules
   - Temporal patterns suggest need for adaptive thresholds

## Future Improvements
- Implement real-time feature extraction pipeline
- Develop specialized models for high-risk merchant categories
- Incorporate network analysis to detect fraud rings
- Deploy online learning system to adapt to evolving fraud patterns
- Implement explainable AI techniques for better investigation support

## Requirements
Major dependencies include:
- pandas>=1.3.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- lightgbm>=3.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- plotly>=5.3.0
- imbalanced-learn>=0.8.0

See `requirements.txt` for a complete list of dependencies.