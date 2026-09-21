"""
Task 4: Machine Learning Model Definitions for Medical Diagnosis
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_clinical_models():
    """
    Returns dictionary of machine learning classifiers evaluated for medical disease prediction.
    """
    return {
        "SVM_RBF": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient_Boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }
