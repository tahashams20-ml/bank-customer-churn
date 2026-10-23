Churn Prediction — Bank Customer Churn Engine

A full-stack machine learning web app that predicts whether a bank customer is likely to leave (churn), powered by a tuned CatBoost classifier. Enter customer attributes and get a real-time churn probability, explore feature importances, and review honest holdout evaluation metrics.

Live Demo

Customer Churn Prediction

Features
Real-time inference: enter customer details and instantly get a churn probability.
Model feature importance: interactive bar chart and ranked table computed directly from the trained CatBoost model.
Holdout evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix on 2,000 unseen test samples.
Dataset Info page: dataset specification, demographic distributions, feature definitions, summary statistics and a raw data preview.
Modern dark UI: responsive, clean dashboard-style interface.
Model Performance (Holdout Test Set, 2,000 samples)

Metrics for the churn class are computed at the tuned decision threshold of 0.6313.

Metric	Score
Accuracy	85.40%
Precision	63.79%
Recall	65.36%
F1-Score	64.56%
ROC-AUC	87.21%

Classification Report

Class	Precision	Recall	F1-Score	Support
Retained (0)	0.91	0.91	0.91	1593
Churned (1)	0.64	0.65	0.65	407
Accuracy			0.85	2000
Macro avg	0.77	0.78	0.78	2000
Weighted avg	0.86	0.85	0.85	2000

Confusion Matrix

	Predicted Retain (0)	Predicted Churn (1)
Actual Retain (0)	1442 (TN)	151 (FP)
Actual Churn (1)	141 (FN)	266 (TP)

The decision threshold was tuned to maximize the F1-score of the churn class, balancing precision and recall rather than favoring one over the other.

Top Feature Importances
Rank	Feature	Importance
1	NumOfProducts	26.51%
2	Age	18.78%
3	Geography	8.69%
4	Balance	7.89%
5	Balance_salary_ratio	5.77%
6	CreditScore_Age_ratio	4.89%
7	EstimatedSalary	4.77%
Dataset
Source file: Churn_Modelling.csv
Records: 10,000 bank customers
Raw columns: 14 → Trained features: 16 (after feature engineering)
Target: Exited (1 = churned, 0 = retained)
Class balance: 79.63% retained / 20.37% churned

Geography: France 50.14% · Germany 25.09% · Spain 24.77% Gender: Male 54.57% · Female 45.43%

Engineered Features
Feature	Description
Balance_flag	1 if account balance is zero
Balance_salary_ratio	log1p(Balance / EstimatedSalary)
Age_group	Young Adults / Middle aged / Senior
CreditScore_Age_ratio	CreditScore / Age
Tenure_Age_ratio	Tenure / Age
IsActive_NumOfProducts	IsActiveMember * NumOfProducts
Input Fields in the App
Field	Range / Values
Credit Score	350 – 850
Geography	France, Germany, Spain
Gender	Male, Female
Age	18 – 100
Tenure	0 – 10 years
Account Balance	Numeric ($)
Number of Products	1 – 4
Has Credit Card	Yes / No
Is Active Member	Yes / No
Estimated Salary	Numeric ($)
Tech Stack
Model: CatBoost Classifier
Data / ML: Python, Pandas, NumPy, Scikit-learn
Frontend: HTML / CSS / JavaScript
Deployment: Vercel
