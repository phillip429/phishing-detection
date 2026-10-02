from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "phishing_model.joblib"
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_names = bundle["feature_names"]
threshold = float(bundle["threshold"])
scaler = bundle.get("scaler")

@app.get("/")
def home():
    return jsonify({
        "service": "Phishing Detection API",
        "status": "running",
        "model": bundle.get("model_name", "unknown"),
        "feature_count": len(feature_names),
        "threshold": threshold
    })

@app.get("/health")
def health():
    return jsonify({"status": "healthy"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object containing the 48 model features."
        }), 400

    missing = [f for f in feature_names if f not in data]
    extra = [f for f in data if f not in feature_names]

    if missing:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing
        }), 400

    try:
        x = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)
        x = x.apply(pd.to_numeric, errors="raise")

        if scaler is not None:
            x_input = scaler.transform(x)
        else:
            x_input = x

        probability = float(model.predict_proba(x_input)[0, 1])
        alert = probability >= threshold

        if probability >= max(threshold, 0.85):
            severity = "high"
        elif probability >= threshold:
            severity = "medium"
        elif probability >= threshold * 0.5:
            severity = "low"
        else:
            severity = "none"

        return jsonify({
            "prediction": "phishing" if alert else "legitimate",
            "phishing_probability": round(probability, 4),
            "alert": bool(alert),
            "severity": severity,
            "threshold": threshold,
            "model": bundle.get("model_name", "unknown"),
            "ignored_extra_features": extra
        })

    except Exception as exc:
        return jsonify({
            "error": "Invalid feature values. All 48 features must be numeric.",
            "details": str(exc)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
