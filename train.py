import time
import joblib
from preprocess import load_data, preprocess
from model import build_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_precision_recall


def main():
    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    print("Loading dataset...")
    df = load_data('creditcard.csv')

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess(df)

    # ── 2. Build & train model ────────────────────────────────────────────────
    print("\nTraining ensemble model (Logistic Regression + Gradient Boosting)...")
    model = build_model()

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.2f}s")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)

    # ── 4. Inference speed test ───────────────────────────────────────────────
    print("\nMeasuring inference speed (single sample)...")
    sample = X_test.iloc[[0]]
    start  = time.time()
    model.predict(sample)
    latency = (time.time() - start) * 1000
    print(f"Inference latency: {latency:.2f}ms")

    # ── 5. Save plots ─────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)
    plot_precision_recall(y_test, y_pred_prob)

    # ── 6. Save model ─────────────────────────────────────────────────────────
    joblib.dump(model, 'fraud_model.pkl')
    print("\nModel saved as fraud_model.pkl")


if __name__ == '__main__':
    main()
