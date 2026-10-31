import os
import joblib
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# ==========================================
# 1. REGRESSION: California Housing
# ==========================================
print("--- Training California Housing Regression Model ---")
housing = fetch_california_housing(as_frame=True)

# Select only 4 features as specified
selected_features = ["MedInc", "HouseAge", "AveRooms", "Population"]
X_reg = housing.data[selected_features]
y_reg = housing.target

# Split data (80% train, 20% test)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

# Evaluate model performance
y_pred_r = reg_model.predict(X_test_r)
r2 = r2_score(y_test_r, y_pred_r)
print(f"California Housing R² Score: {r2:.4f}")

# Save regression model
joblib.dump(reg_model, "models/regression.joblib")
print("Saved: models/regression.joblib\n")

# ==========================================
# 2. CLASSIFICATION: Iris Dataset
# ==========================================
print("--- Training Iris Classification Model ---")
iris = load_iris()
X_cls = iris.data
y_cls = iris.target

# Split data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Train Logistic Regression model
cls_model = LogisticRegression(max_iter=200, random_state=42)
cls_model.fit(X_train_c, y_train_c)

# Evaluate model performance
y_pred_c = cls_model.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print(f"Iris Classification Accuracy: {acc:.4f}")

# Save classification model
joblib.dump(cls_model, "models/classification.joblib")
print("Saved: models/classification.joblib\n")