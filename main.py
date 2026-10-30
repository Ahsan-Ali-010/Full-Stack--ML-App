import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import (
    HousingInput, HousingOutput, 
    ClassificationInput, ClassificationOutput,
    TextInput, TextOutput
)

app = FastAPI(title="ML Full-Stack API", version="2.0")

# Clean CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "reg": "models/regression.joblib",
    "cls": "models/classification.joblib",
    "nlp": "models/sentiment.joblib"
}

models = {}

@app.on_event("startup")
def load_models():
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required model file: {path}. Run train.py first.")
        models[name] = joblib.load(path)

@app.get("/")
def read_root():
    return {"status": "Online", "models_loaded": list(models.keys())}

@app.post("/predict/regression", response_model=HousingOutput)
def predict_regression(data: HousingInput):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        raw_val = float(models["reg"].predict(input_df)[0])
        price_usd = round(raw_val * 100000, 2)
        
        return HousingOutput(
            predicted_raw_val=round(raw_val, 4),
            price_usd=price_usd,
            formatted_price=f"${price_usd:,.2f}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/classification", response_model=ClassificationOutput)
def predict_classification(data: ClassificationInput):
    try:
        input_df = pd.DataFrame([{
            'sepal length (cm)': data.sepal_length,
            'sepal width (cm)': data.sepal_width,
            'petal length (cm)': data.petal_length,
            'petal width (cm)': data.petal_width
        }])
        
        prob = float(models["cls"].predict_proba(input_df)[0][1])
        is_event = prob >= data.threshold
        
        return ClassificationOutput(
            is_rush_event=is_event,
            probability=round(prob, 4),
            threshold_used=data.threshold,
            status="Rush Event Flagged" if is_event else "Normal Operation"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sentiment", response_model=TextOutput)
def predict_sentiment(data: TextInput):
    try:
        probs = models["nlp"].predict_proba([data.text])[0]
        pos_prob = float(probs[1])
        sentiment = "Positive" if pos_prob >= 0.5 else "Negative"
        confidence = pos_prob if sentiment == "Positive" else (1.0 - pos_prob)
        
        return TextOutput(
            sentiment=sentiment,
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))