import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test):
    """Print full evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    print("\n========== Classification Report ==========")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC-ROC Score: {auc:.4f}")

    return y_pred, y_pred_prob


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("Confusion matrix saved as confusion_matrix.png")


def plot_roc_curve(y_test, y_pred_prob):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Credit Card Fraud Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()
    print("ROC curve saved as roc_curve.png")


def plot_precision_recall(y_test, y_pred_prob):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=150)
    plt.show()
    print("Precision-recall curve saved as precision_recall_curve.png")
