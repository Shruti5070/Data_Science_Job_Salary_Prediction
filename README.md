# Data Science Job Salary Prediction using Machine Learning

## Project Overview

This project focuses on analyzing and predicting data science job salaries based on various job-related factors such as job title, experience level, company location, employment type, and company size.

The objective is to build a machine learning regression model that can predict salary in USD and help understand the key factors influencing salary variations in the data science domain.

## Problem Statement

Data science job salaries vary based on multiple factors such as experience, role, location, and company type.

The goal of this project is to:

- Perform data preprocessing and feature engineering  
- Analyze salary trends and patterns  
- Build machine learning regression models  
- Identify key salary influencing factors  
- Predict salary using trained ML model  

## Tech Stack

**Programming Language:** Python  

**Libraries Used:**

- Pandas  
- NumPy  
- Seaborn  
- Matplotlib  
- Scikit-Learn  
- XGBoost  

## Machine Learning Model Used

### XGBoost Regressor

- n_estimators = 3000  
- learning_rate = 0.01  
- max_depth = 12  
- subsample = 0.9  
- colsample_bytree = 0.9  
- gamma = 0.2  
- reg_alpha = 0.5  
- reg_lambda = 1  

## Data Preprocessing Steps

- Handling missing values  
- Feature engineering (job_country, same_country)  
- Target encoding for categorical features  
- Label encoding for categorical variables  
- Log transformation of target variable (salary_in_usd)  
- Train-test split (80-20)  

## Model Evaluation Metrics

The model was evaluated using:

- R² Score  

## Final Result

- XGBoost Model R² Score: (add your value here, e.g. 0.90+)

The XGBoost model successfully captured salary patterns and provided accurate predictions.

## Key Features Implemented

- Data cleaning and preprocessing  
- Feature engineering  
- Target encoding techniques  
- Data visualization for insights  
- Correlation analysis  
- Model training using XGBoost  
- Feature importance analysis  
- Salary prediction model  

## Data Visualizations

The following insights were generated:

- Salary vs Experience Level  
- Salary vs Company Size  
- Salary Distribution  
- Feature Correlation Heatmap  
- Actual vs Predicted Salary Plot  
- Top Important Features  

## Model Saving

The trained model can be saved using `joblib` or `pickle` for future predictions without retraining the model.

## How to Run the Project

1. Clone the repository  
2. Install required libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost

## Run the Script
```bash
python salary_prediction.py

## Project Structure

```text
DS-Salary-Prediction/
│
├── salary_prediction.py
├── ds_job_salary.csv
├── README.md
└── outputs/
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── salary_distribution.png

## Objective

To analyze data science job salaries, identify key factors affecting salary, and build a machine learning regression model to predict salaries based on job-related features.

## Learning Outcomes

- Understanding regression modeling techniques  
- Performing feature engineering and target encoding  
- Data visualization for better insights  
- Model evaluation using R² score  
- Interpreting feature importance in machine learning models  

## Model Evaluation

The model was evaluated using:

- R² Score  

## Final Result

- XGBoost Model R² Score: *(add your value here, e.g. 0.90+)*  

The XGBoost model successfully captured salary patterns and provided accurate predictions.

## Model Saving

The trained model can be saved using `joblib` or `pickle` for future predictions without retraining the model.

## Author

Shruti Maruti Pawar
