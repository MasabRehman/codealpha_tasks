"""
Task 1: Inference & Scoring Engine for Credit Scoring Model
Provides credit score calculation (300-850), default risk probability, and credit decisioning.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add base directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.feature_engineering import probability_to_credit_score, get_credit_rating

def load_model():
    model_path = os.path.join(base_dir, "models", "credit_scoring_best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
    return joblib.load(model_path)

def score_applicant(applicant_data: dict, model=None) -> dict:
    """
    Takes an applicant's financial attributes dictionary and returns:
    - Default Probability
    - FICO Credit Score (300-850)
    - Rating Tier & Risk Level
    - Approval Recommendation
    - Key Risk Factor Breakdown
    """
    if model is None:
        model = load_model()
        
    df_applicant = pd.DataFrame([applicant_data])
    
    # Predict default probability
    if hasattr(model, "predict_proba"):
        prob_default = float(model.predict_proba(df_applicant)[0, 1])
    else:
        prob_default = 0.5
        
    credit_score = probability_to_credit_score(prob_default)
    rating_info = get_credit_rating(credit_score)
    
    # Identify key risk factors
    risk_factors = []
    if applicant_data.get("Credit_Utilization_Ratio", 0) > 45:
        risk_factors.append(f"High credit utilization ({applicant_data['Credit_Utilization_Ratio']}%) - ideally under 30%")
    if applicant_data.get("Outstanding_Debt", 0) > applicant_data.get("Annual_Income", 1) * 0.4:
        risk_factors.append("High debt burden relative to annual income")
    if applicant_data.get("Num_of_Delayed_Payment", 0) > 3:
        risk_factors.append(f"Frequent past payment delays ({applicant_data['Num_of_Delayed_Payment']} recorded)")
    if applicant_data.get("Delay_from_due_date", 0) > 15:
        risk_factors.append(f"Significant overdue delay days ({applicant_data['Delay_from_due_date']} days)")
    if applicant_data.get("Num_Credit_Inquiries", 0) > 5:
        risk_factors.append(f"Multiple recent hard credit inquiries ({applicant_data['Num_Credit_Inquiries']})")
    if applicant_data.get("Payment_of_Min_Amount") == "Yes":
        risk_factors.append("Paying only minimum monthly dues (signals potential cash flow stress)")
    if applicant_data.get("Credit_Mix") == "Bad":
        risk_factors.append("Suboptimal credit mix across loan and card types")
        
    if not risk_factors:
        risk_factors.append("Clean financial profile with healthy credit utilization and on-time payment track record.")
        
    return {
        "Applicant_Profile": applicant_data,
        "Probability_of_Default": round(prob_default, 4),
        "Calculated_Credit_Score": credit_score,
        "Rating_Tier": rating_info["tier"],
        "Risk_Level": rating_info["risk_level"],
        "Recommendation": rating_info["approval_recommendation"],
        "Risk_Factors": risk_factors
    }

if __name__ == "__main__":
    print("=" * 65)
    print("      TASK 1: CREDIT SCORING & DECISIONING ENGINE")
    print("=" * 65)
    
    # Test with Sample Applicant 1: Prime Profile
    sample_prime = {
        "Age": 38,
        "Annual_Income": 95000.0,
        "Monthly_Inhand_Salary": 6200.0,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 3,
        "Num_of_Loan": 1,
        "Delay_from_due_date": 1,
        "Num_of_Delayed_Payment": 0,
        "Num_Credit_Inquiries": 1,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 12000.0,
        "Credit_Utilization_Ratio": 18.5,
        "Credit_History_Age_Months": 180,
        "Payment_of_Min_Amount": "No",
        "Total_EMI_per_month": 450.0,
        "Monthly_Balance": 3800.0,
        "Payment_Behaviour": "High_spent_Large_value_payments"
    }
    
    # Test with Sample Applicant 2: Subprime / High Risk Profile
    sample_subprime = {
        "Age": 24,
        "Annual_Income": 28000.0,
        "Monthly_Inhand_Salary": 1900.0,
        "Num_Bank_Accounts": 6,
        "Num_Credit_Card": 8,
        "Num_of_Loan": 4,
        "Delay_from_due_date": 25,
        "Num_of_Delayed_Payment": 8,
        "Num_Credit_Inquiries": 7,
        "Credit_Mix": "Bad",
        "Outstanding_Debt": 22000.0,
        "Credit_Utilization_Ratio": 78.5,
        "Credit_History_Age_Months": 24,
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 750.0,
        "Monthly_Balance": 120.0,
        "Payment_Behaviour": "Low_spent_Small_value_payments"
    }
    
    try:
        model = load_model()
        
        print("\n[Case 1: Prime Applicant Assessment]")
        res1 = score_applicant(sample_prime, model)
        print(f"Credit Score:   {res1['Calculated_Credit_Score']} / 850 ({res1['Rating_Tier']})")
        print(f"Default Risk:   {res1['Probability_of_Default']*100:.1f}%")
        print(f"Decision:       {res1['Recommendation']}")
        print("Key Factors:    " + ", ".join(res1['Risk_Factors']))
        
        print("\n[Case 2: High Risk Applicant Assessment]")
        res2 = score_applicant(sample_subprime, model)
        print(f"Credit Score:   {res2['Calculated_Credit_Score']} / 850 ({res2['Rating_Tier']})")
        print(f"Default Risk:   {res2['Probability_of_Default']*100:.1f}%")
        print(f"Decision:       {res2['Recommendation']}")
        print("Key Factors:    " + ", ".join(res2['Risk_Factors']))
        print("=" * 65)
    except Exception as e:
        print(f"Note: Model needs training first: {e}")
