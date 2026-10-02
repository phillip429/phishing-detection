# Phishing Detection API

Deployment of the trained Random Forest phishing model using Flask and Render.

## Model
- Algorithm: Random Forest
- Calibrated probabilities: Isotonic calibration
- Features: 48
- Alert threshold: `0.03787940050676586`
- Dataset used during training: 10,000 rows, 48 features, balanced classes

## Endpoints

### GET /
Returns API status and model information.

### GET /health
Health check for Render.

### POST /predict
Accepts a JSON object containing all 48 trained features.

The API returns:
- prediction
- phishing probability
- alert status
- severity
- threshold

## Important
The trained model expects pre-extracted URL/page features. It does not extract these 48 features from a raw URL by itself.
