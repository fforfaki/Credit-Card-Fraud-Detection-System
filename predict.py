import joblib
import pandas as pd
import time


def predict(input_data: dict):
    """
    Run fraud prediction on a single transaction.

    Args:
        input_data: dict with keys matching the creditcard.csv feature columns

    Returns:
        prediction (0 = Legit, 1 = Fraud) and confidence score
    """
    model = joblib.load('fraud_model.pkl')

    df = pd.DataFrame([input_data])

    start      = time.time()
    prediction = model.predict(df)[0]
    confidence = model.predict_proba(df)[0][prediction]
    latency    = (time.time() - start) * 1000

    label = "FRAUD" if prediction == 1 else "LEGIT"
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Latency    : {latency:.2f}ms")

    return prediction, confidence


if __name__ == '__main__':
    # Example — replace with real feature values
    sample = {f'V{i}': 0.0 for i in range(1, 29)}
    sample['Time']   = 0.0
    sample['Amount'] = 0.0
    predict(sample)
