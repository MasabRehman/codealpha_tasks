"""
Task 1: Credit Scoring Models Definition & Training Factory
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier

def get_candidate_models():
    """
    Returns a dictionary of candidate classification algorithms for credit scoring.
    """
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        )
    }
    return models
