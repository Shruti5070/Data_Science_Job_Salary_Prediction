#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("ds_job_salary.csv")
df.head()

df.shape
df.info()
df.describe()

# Check missing values
df.isnull().sum()

# Feature engineering
df["job_country"] = df["job_title"] + "_" + df["company_location"]
df["same_country"] = (df["employee_residence"] == df["company_location"]).astype(int)

# Target encoding
job_mean = df.groupby("job_title")["salary_in_usd"].mean()
df["job_encoded"] = df["job_title"].map(job_mean)

job_country_mean = df.groupby("job_country")["salary_in_usd"].mean()
df["job_country_encoded"] = df["job_country"].map(job_country_mean)

loc_mean = df.groupby("company_location")["salary_in_usd"].mean()
df["company_encoded"] = df["company_location"].map(loc_mean)

# Encoding categorical variables
df["experience_level"] = df["experience_level"].map({
    "EN": 1, "MI": 2, "SE": 3, "EX": 4
})

df["company_size"] = df["company_size"].map({
    "S": 1, "M": 2, "L": 3
})

df["employment_type"] = df["employment_type"].map({
    "PT": 1, "FT": 2, "CT": 3, "FL": 4
})

# Drop columns
df = df.drop(columns=[
    "job_title",
    "employee_residence",
    "company_location",
    "job_country",
    "salary_currency"
])

# Log transform target
df["salary_in_usd"] = np.log1p(df["salary_in_usd"])

# Define features and target
X = df.drop("salary_in_usd", axis=1)
y = df["salary_in_usd"]

print(X.dtypes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=12,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, pred))

# Visualization
plt.figure()
sns.boxplot(x="experience_level", y="salary_in_usd", data=df)
plt.title("Salary vs Experience Level")
plt.show()

plt.figure()
sns.boxplot(x="company_size", y="salary_in_usd", data=df)
plt.title("Salary vs Company Size")
plt.show()

plt.figure()
sns.histplot(df["salary_in_usd"], bins=50)
plt.title("Salary Distribution")
plt.show()

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

plt.figure()
plt.scatter(y_test, pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.show()

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=imp_df.head(10))
plt.title("Top Features")
plt.show()