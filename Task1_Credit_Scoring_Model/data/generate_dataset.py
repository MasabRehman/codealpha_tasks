"""
Task 1: Credit Scoring Model - Dataset Generator
Generates realistic financial data simulating credit bureau and banking records.
"""

import os
import numpy as np
import pandas as pd

def generate_credit_data(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic credit scoring dataset.
    Features include income, debt, payment history, credit utilization, and credit mix.
    Target: 'Credit_Default' (1 = Default / High Risk, 0 = Good Standing)
    """
    np.random.seed(random_state)
    
    # 1. Demographic & Basic Financials
    age = np.random.randint(18, 70, size=n_samples)
    annual_income = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples) # approx 20k - 250k
    annual_income = np.clip(annual_income, 15000, 300000).round(2)
    monthly_inhand_salary = (annual_income / 12 * np.random.uniform(0.72, 0.85, size=n_samples)).round(2)
    
    # 2. Banking & Credit Lines
    num_bank_accounts = np.random.poisson(lam=3, size=n_samples) + 1
    num_credit_cards = np.random.poisson(lam=4, size=n_samples) + 1
    num_of_loans = np.random.poisson(lam=2, size=n_samples)
    
    # 3. Credit Utilization & Debt
    credit_utilization_ratio = np.random.beta(a=2, b=5, size=n_samples) * 100 # percentage (0 - 100%)
    outstanding_debt = (annual_income * np.random.beta(a=2, b=4, size=n_samples) * 0.8).round(2)
    total_emi_per_month = (outstanding_debt / np.maximum(num_of_loans * 12, 12) * np.random.uniform(0.8, 1.2, size=n_samples)).round(2)
    total_emi_per_month = np.clip(total_emi_per_month, 0, monthly_inhand_salary * 0.7).round(2)
    
    # 4. Payment History & Delinquencies
    credit_history_age_months = (age - 18) * 12 * np.random.uniform(0.2, 0.9, size=n_samples)
    credit_history_age_months = np.clip(credit_history_age_months, 6, 600).astype(int)
    
    delay_from_due_date = np.random.exponential(scale=5, size=n_samples).astype(int)
    num_of_delayed_payment = (delay_from_due_date / 4 + np.random.poisson(lam=1.5, size=n_samples)).astype(int)
    num_credit_inquiries = np.random.poisson(lam=3, size=n_samples)
    
    payment_of_min_amount = np.random.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    
    # 5. Categorical Credit Profile
    credit_mix = np.random.choice(["Good", "Standard", "Bad"], size=n_samples, p=[0.35, 0.45, 0.20])
    payment_behaviour = np.random.choice(
        [
            "High_spent_Small_value_payments",
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "Low_spent_Large_value_payments"
        ],
        size=n_samples
    )
    
    monthly_balance = np.maximum(50.0, monthly_inhand_salary - total_emi_per_month - (monthly_inhand_salary * np.random.uniform(0.3, 0.6, size=n_samples))).round(2)
    
    # 6. Synthesize Realistic Risk Score & Default Label
    credit_mix_score = pd.Series(credit_mix).map({"Good": -1.2, "Standard": 0.0, "Bad": 1.5}).values
    min_amt_score = (payment_of_min_amount == "Yes").astype(float) * 0.8
    
    z = (
        - 0.000018 * annual_income
        + 0.045 * credit_utilization_ratio
        + 0.000035 * outstanding_debt
        + 0.09 * delay_from_due_date
        + 0.14 * num_of_delayed_payment
        + 0.16 * num_credit_inquiries
        - 0.006 * credit_history_age_months
        + credit_mix_score
        + min_amt_score
        + 0.00045 * total_emi_per_month
        - 0.00025 * monthly_balance
        - 1.4
    )
    
    default_prob = 1 / (1 + np.exp(-z))
    credit_default = (np.random.rand(n_samples) < default_prob).astype(int)
    
    credit_score_category = []
    for p in default_prob:
        if p < 0.25:
            credit_score_category.append("Good")
        elif p < 0.60:
            credit_score_category.append("Standard")
        else:
            credit_score_category.append("Poor")
            
    df = pd.DataFrame({
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_inhand_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Card": num_credit_cards,
        "Num_of_Loan": num_of_loans,
        "Delay_from_due_date": delay_from_due_date,
        "Num_of_Delayed_Payment": num_of_delayed_payment,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization_ratio.round(2),
        "Credit_History_Age_Months": credit_history_age_months,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Total_EMI_per_month": total_emi_per_month,
        "Monthly_Balance": monthly_balance,
        "Payment_Behaviour": payment_behaviour,
        "Credit_Score_Category": credit_score_category,
        "Credit_Default": credit_default
    })
    
    return df

def save_default_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(current_dir, exist_ok=True)
    output_path = os.path.join(current_dir, "credit_scoring_dataset.csv")
    df = generate_credit_data(n_samples=6000, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print("Default Class distribution:")
    print(df['Credit_Default'].value_counts(normalize=True))
    return output_path

if __name__ == "__main__":
    save_default_dataset()
