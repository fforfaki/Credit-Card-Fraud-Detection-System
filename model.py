from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier


def build_model():
    """
    Ensemble model combining Logistic Regression and Gradient Boosting
    using soft voting for fraud detection.
    """
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('gb', gb)],
        voting='soft'
    )

    return ensemble
