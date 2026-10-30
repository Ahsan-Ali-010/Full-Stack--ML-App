import os
import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

os.makedirs("models", exist_ok=True)

# =========================================================
# 1. REGRESSION COMPARISON (Lasso vs. Ridge)
# =========================================================
print("=== Training Regression Candidates ===")
data_housing = fetch_california_housing(as_frame=True)
X_reg = data_housing.data[['MedInc', 'HouseAge', 'AveRooms', 'Population']]
y_reg = data_housing.target

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_candidates = {
    "Lasso": Lasso(alpha=0.01, random_state=42),
    "Ridge": Ridge(alpha=1.0, random_state=42)
}

best_reg_score = -float("inf")
best_reg_pipe = None
best_reg_name = ""

for name, model in reg_candidates.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_tr_r, y_tr_r)
    score = r2_score(y_te_r, pipe.predict(X_te_r))
    print(f" -> {name} R² Score: {score:.4f}")
    
    if score > best_reg_score:
        best_reg_score = score
        best_reg_pipe = pipe
        best_reg_name = name

print(f"Saving best regression model ({best_reg_name}) -> models/regression.joblib\n")
joblib.dump(best_reg_pipe, "models/regression.joblib")

# =========================================================
# 2. CLASSIFICATION COMPARISON (Random Forest vs. Gradient Boosting)
# =========================================================
print("=== Training Classification Candidates ===")
data_iris = load_iris(as_frame=True)
X_cls = data_iris.data
y_cls = (data_iris.target == 2).astype(int)

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

cls_candidates = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

best_cls_score = -float("inf")
best_cls_pipe = None
best_cls_name = ""

for name, model in cls_candidates.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipe.fit(X_tr_c, y_tr_c)
    auc = roc_auc_score(y_te_c, pipe.predict_proba(X_te_c)[:, 1])
    print(f" -> {name} ROC-AUC: {auc:.4f}")
    
    if auc > best_cls_score:
        best_cls_score = auc
        best_cls_pipe = pipe
        best_cls_name = name

print(f"Saving best classification model ({best_cls_name}) -> models/classification.joblib\n")
joblib.dump(best_cls_pipe, "models/classification.joblib")

# =========================================================
# 3. NLP SENTIMENT MODEL
# =========================================================
print("=== Training Sentiment NLP Model ===")
reviews = [
    "The coffee was cold and service was slow", "Excellent food and great atmosphere",
    "Rude waiter and dirty table", "Loved the flat white, highly recommended",
    "Overpriced, disappointed with delivery", "Wonderful experience and friendly staff"
]
labels = [0, 1, 0, 1, 0, 1]

nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('model', GradientBoostingClassifier(random_state=42))
])
nlp_pipeline.fit(reviews, labels)
joblib.dump(nlp_pipeline, "models/sentiment.joblib")

print("All top-performing models saved successfully!")