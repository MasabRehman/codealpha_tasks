"""
Task 1: Feature Engineering & Preprocessing Pipeline for Credit Scoring
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer financial ratios and credit risk metrics.
    """
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        
        # 1. Debt-to-Income Ratio (DTI)
        df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1e-5)
        
        # 2. Payment-to-Income Ratio (PTI)
        df["Payment_to_Income_Ratio"] = (df["Total_EMI_per_month"] * 12) / (df["Annual_Income"] + 1e-5)
        
        # 3. Delinquency Risk Severity Index
        df["Delinquency_Severity"] = df["Delay_from_due_date"] * (df["Num_of_Delayed_Payment"] + 1)
        
        # 4. Savings / Surplus Ratio
        df["Savings_Ratio"] = df["Monthly_Balance"] / (df["Monthly_Inhand_Salary"] + 1e-5)
        
        # 5. Credit Line Density
        df["Credit_Line_Density"] = df["Num_Credit_Card"] / (df["Num_Bank_Accounts"] + 1e-5)
        
        # 6. Credit History Experience Ratio
        df["Credit_History_Experience"] = df["Credit_History_Age_Months"] / (df["Age"] * 12 + 1e-5)
        
        # 7. Total Inquiries & Delays burden
        df["Inquiry_Delinquency_Burden"] = df["Num_Credit_Inquiries"] * 1.5 + df["Num_of_Delayed_Payment"]
        
        return df

def get_preprocessor():
    """
    Returns an integrated preprocessing pipeline with custom feature engineering,
    imputation, categorical one-hot encoding, and numerical standardization.
    """
    num_features = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age_Months", "Total_EMI_per_month", "Monthly_Balance",
        "Debt_to_Income_Ratio", "Payment_to_Income_Ratio", "Delinquency_Severity",
        "Savings_Ratio", "Credit_Line_Density", "Credit_History_Experience",
        "Inquiry_Delinquency_Burden"
    ]
    
    cat_features = [
        "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"
    ]
    
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    column_preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ])
    
    full_pipeline = Pipeline(steps=[
        ("feature_engineer", CreditFeatureEngineer()),
        ("preprocessor", column_preprocessor)
    ])
    
    return full_pipeline

def probability_to_credit_score(default_prob: float) -> int:
    """
    Converts default probability into a standard FICO-scale Credit Score (300 - 850).
    Higher score = Better creditworthiness (Lower default probability).
    """
    prob = np.clip(default_prob, 0.0, 1.0)
    score = int(round(850 - (550 * prob)))
    return int(np.clip(score, 300, 850))

def get_credit_rating(score: int) -> dict:
    """
    Returns the qualitative rating tier, risk assessment, and recommendation based on credit score.
    """
    if score >= 750:
        return {
            "tier": "Excellent",
            "risk_level": "Low Risk",
            "approval_recommendation": "Instant Approval - Premium Tier",
            "badge_color": "green"
        }
    elif score >= 700:
        return {
            "tier": "Good",
            "risk_level": "Low-to-Moderate Risk",
            "approval_recommendation": "Approved - Standard Rates",
            "badge_color": "blue"
        }
    elif score >= 650:
        return {
            "tier": "Fair",
            "risk_level": "Moderate Risk",
            "approval_recommendation": "Conditional Approval - Additional Verification",
            "badge_color": "orange"
        }
    elif score >= 550:
        return {
            "tier": "Poor",
            "risk_level": "High Risk",
            "approval_recommendation": "High Risk - Requires Collateral / Co-signer",
            "badge_color": "red"
        }
    else:
        return {
            "tier": "Very Poor",
            "risk_level": "Severe Risk",
            "approval_recommendation": "Decline - High Likelihood of Default",
            "badge_color": "darkred"
        }
