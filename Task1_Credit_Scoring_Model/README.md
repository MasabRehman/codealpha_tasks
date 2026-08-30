# ✅ Task 1: Credit Scoring & Default Risk Assessment Model

## 📌 Project Overview
The **Credit Scoring Model** predicts an individual's creditworthiness and default risk using historical financial, demographic, and banking records. It transforms machine learning default probabilities into standard **FICO-scale credit scores (300–850)** and provides explainable underwriting decisions (Instant Approval, Conditional Approval, or Decline).

---

## 🏗️ Architecture & Features

1. **Feature Engineering**:
   - **Debt-to-Income Ratio (DTI)**: Measures total debt relative to annual earnings.
   - **Payment-to-Income Ratio (PTI)**: Evaluates monthly EMI obligations.
   - **Delinquency Severity Index**: Penalizes overdue delay days multiplied by missed payment frequencies.
   - **Credit Card Utilization**: Tracks revolving credit saturation.
   - **Savings & Balance Ratios**: Quantifies net cash flow liquidity.

2. **Classification Algorithms Evaluated**:
   - **Logistic Regression** (L2 Regularized with balanced class weighting)
   - **Decision Tree Classifier** (Constrained depth for interpretability)
   - **Random Forest Classifier** (Ensemble of bagged trees)
   - **Gradient Boosting Classifier** (HistGradientBoosting / LightGBM-style optimization)

3. **Evaluation Metrics**:
   - **ROC-AUC & PR-AUC** (Primary discrimination metrics)
   - **Precision, Recall & F1-Score**
   - **Kolmogorov-Smirnov (KS) Statistic** (Banking bureau industry standard)
   - **Confusion Matrix**

4. **Scoring Engine**:
   - Continuous score mapping: $\text{Score} = \text{round}(850 - 550 \times P(\text{Default})) \in [300, 850]$
   - Rating Tiers: Excellent ($\ge 750$), Good ($700-749$), Fair ($650-699$), Poor ($550-649$), Very Poor ($<550$).

---

## 📂 Directory Structure

```
Task1_Credit_Scoring_Model/
├── data/
│   ├── generate_dataset.py      # Synthetic financial dataset generator
│   └── credit_scoring_dataset.csv # 6,000 applicant financial records
├── src/
│   ├── feature_engineering.py   # Transformer pipelines, ratio calculation & FICO mapping
│   └── models.py                # Model configurations & factory
├── models/
│   └── credit_scoring_best_model.joblib # Saved production pipeline artifact
├── results/
│   ├── evaluation_curves.png    # ROC, PR, and Confusion Matrix plots
│   └── model_comparison.json    # Benchmark metrics comparison table
├── train.py                     # Training & benchmark script
├── evaluate.py                  # Evaluation & visualization script
├── predict.py                   # Single applicant & batch CLI scoring engine
├── app.py                       # Interactive Streamlit Web App
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quickstart & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models & Run Benchmark
```bash
python train.py
```

### 3. Evaluate & Generate Plots
```bash
python evaluate.py
```

### 4. Run CLI Prediction on Sample Applicants
```bash
python predict.py
```

### 5. Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
