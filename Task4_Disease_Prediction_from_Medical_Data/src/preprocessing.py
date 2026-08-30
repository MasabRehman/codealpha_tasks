"""
Task 4: Clinical Preprocessing & Feature Engineering Module for Medical Data
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class MedicalDataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans physiological invalid zeroes in clinical measurements (e.g. Glucose=0 or BP=0).
    """
    def __init__(self, zero_as_nan_cols=None):
        self.zero_as_nan_cols = zero_as_nan_cols or ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        df = X.copy()
        if isinstance(df, pd.DataFrame):
            for col in self.zero_as_nan_cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
        return df

def get_medical_pipeline(model):
    """
    Wraps model into a standardized pipeline with cleaner, median imputer and scaler.
    """
    pipeline = Pipeline(steps=[
        ("cleaner", MedicalDataCleaner()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    return pipeline

def stratify_medical_risk(probability: float) -> dict:
    """
    Stratifies diagnostic probability into clinical risk categories.
    """
    prob_pct = probability * 100
    if probability < 0.25:
        return {
            "risk_level": "Low Risk",
            "badge_color": "green",
            "risk_percentage": f"{prob_pct:.1f}%",
            "clinical_advice": "Parameters within healthy ranges. Maintain regular annual checkups and balanced lifestyle."
        }
    elif probability < 0.60:
        return {
            "risk_level": "Moderate / Borderline Risk",
            "badge_color": "orange",
            "risk_percentage": f"{prob_pct:.1f}%",
            "clinical_advice": "Elevated risk markers detected. Recommend confirmatory lab testing and lifestyle intervention."
        }
    else:
        return {
            "risk_level": "High Clinical Risk",
            "badge_color": "red",
            "risk_percentage": f"{prob_pct:.1f}%",
            "clinical_advice": "Significant clinical indicators present. Strongly advise immediate physician consultation and formal diagnosis."
        }
