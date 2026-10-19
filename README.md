# Machine Learning Application with FastAPI & Scikit-Learn

This project demonstrates a full-stack Machine Learning application serving two built-in Scikit-Learn datasets via a FastAPI backend and an HTML/JS frontend.

## Datasets & Models

1. **Regression Model (California Housing):**
   - **Features used:** `MedInc`, `HouseAge`, `AveRooms`, `Population`
   - **Model:** `LinearRegression`
   - **Evaluation:** Evaluated using $R^2$ score (typically ~0.50–0.55).

2. **Classification Model (Iris Dataset):**
   - **Features used:** `sepal length`, `sepal width`, `petal length`, `petal width` (4 features)
   - **Model:** `LogisticRegression`
   - **Evaluation:** Evaluated using Accuracy score (typically ~0.95–1.00).

## Project Structure

```text
ml-app/
├── train.py            # Trains models and exports to joblib
├── main.py             # FastAPI backend server
├── index.html          # Web application dashboard
├── models/
│   ├── regression.joblib
│   └── classification.joblib
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
└── .gitignore