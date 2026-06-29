# Bank Customer Churn Prediction — ML Mid Project

## Overview
This project predicts whether a bank customer will churn (leave) or stay, using the Churn_Modelling dataset with 10,000 records and 14 features. It was developed as a mid-semester Machine Learning Lab project.

## Dataset
- Source: Churn_Modelling.csv
- Records: 10,000 customers
- Target variable: Exited (0 = Stay, 1 = Churn)
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

## Project Structure
- EDA: Univariate, Bivariate, and Multivariate analysis with visualizations
- Data Cleaning: Duplicate removal, outlier handling (IQR + Z-score)
- Feature Encoding: Label Encoding, One-Hot Encoding, Ordinal Encoding
- Feature Scaling: MinMax, Standard, Robust Scaler + Log and Quantile Transforms
- Models Applied: Linear Regression, Logistic Regression, Polynomial Regression, Decision Tree (Entropy & Gini), Naive Bayes (Gaussian, Bernoulli, Multinomial), KNN, SVM

## Model Results (Test Accuracy)

| Model                        | Test Accuracy |
|-----------------------------|---------------|
| Linear/Logistic Regression  | 81.10%        |
| Decision Tree (Entropy)     | 84.25%        |
| Decision Tree (Gini)        | 84.25%        |
| Gaussian Naive Bayes        | 82.95%        |
| Bernoulli Naive Bayes       | 81.15%        |
| Multinomial Naive Bayes     | 79.75%        |
| KNN (k=21, Euclidean)       | 85.35%        |
| SVM (RBF, C=1, gamma=0.1)   | 86.55% ← Best |

## Best Model
SVM with RBF kernel achieved the highest test accuracy of 86.55%. It handled the non-linear patterns in the dataset most effectively.

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, ydata-profiling, Google Colab

## Group Members
- Amna Kauser 
- Zarlish Batool 
- Muhammad Abdullah
