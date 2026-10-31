import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="California Housing & Iris Prediction API",
    description="Backend API serving California Housing Regression and Iris Species Classification models.",
)

# Enabling CORS for direct browser connection from local files
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path definitions
REG_PATH = "models/regression.joblib"
CLS_PATH = "models/classification.joblib"

if not os.path.exists(REG_PATH) or not os.path.exists(CLS_PATH):
    raise RuntimeError("Missing model files. Run 'python train.py' first.")

reg_model = joblib.load(REG_PATH)
cls_model = joblib.load(CLS_PATH)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


# Input Data Schemas
class HousingInput(BaseModel):
    MedInc: float = Field(
        ...,
        description="Median income in block group (e.g., 3.8 = $38,000)",
        example=3.83,
    )
    HouseAge: float = Field(
        ..., description="Median house age in years", example=21.0
    )
    AveRooms: float = Field(
        ..., description="Average rooms per household", example=6.2
    )
    Population: float = Field(
        ..., description="Block group population", example=850.0
    )


class IrisInput(BaseModel):
    sepal_length: float = Field(
        ..., description="Sepal length in cm", example=5.1
    )
    sepal_width: float = Field(
        ..., description="Sepal width in cm", example=3.5
    )
    petal_length: float = Field(
        ..., description="Petal length in cm", example=1.4
    )
    petal_width: float = Field(
        ..., description="Petal width in cm", example=0.2
    )


@app.get("/")
def home():
    return {"status": "Active", "message": "ML Model API is live!"}


@app.post("/predict/regression")
def predict_regression(data: HousingInput):
    try:
        features = np.array(
            [[data.MedInc, data.HouseAge, data.AveRooms, data.Population]]
        )
        prediction = reg_model.predict(features)[0]
        # California housing values are expressed in $100,000s
        estimated_usd = float(prediction * 100000)

        return {
            "predicted_raw_val": round(float(prediction), 4),
            "estimated_price_usd": round(estimated_usd, 2),
            "formatted_price": f"${estimated_usd:,.2f}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/classification")
def predict_classification(data: IrisInput):
    try:
        features = np.array(
            [[
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width,
            ]]
        )
        pred_class_id = int(cls_model.predict(features)[0])
        probabilities = cls_model.predict_proba(features)[0].tolist()
        class_name = IRIS_CLASSES[pred_class_id]

        return {
            "class_id": pred_class_id,
            "predicted_species": class_name,
            "probabilities": {
                IRIS_CLASSES[i]: round(probabilities[i], 4)
                for i in range(len(IRIS_CLASSES))
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))