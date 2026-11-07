import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

np.random.seed(42)
n_samples = 1500

applicant_income = np.random.randint(15000, 250000, size=n_samples)
coapplicant_income = np.random.randint(0, 100000, size=n_samples)
loan_amount = np.random.randint(50000, 1500000, size=n_samples)
credit_history = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])
employment_status = np.random.choice([0, 1], size=n_samples, p=[0.20, 0.80])
existing_loans_count = np.random.randint(0, 5, size=n_samples)

data = pd.DataFrame({
    "applicant_income": applicant_income,
    "coapplicant_income": coapplicant_income,
    "loan_amount": loan_amount,
    "credit_history": credit_history,
    "employment_status": employment_status,
    "existing_loans_count": existing_loans_count
})

debt_to_income = data["loan_amount"] / (data["applicant_income"] + data["coapplicant_income"] + 1)
logits = (
    3.2 * data["credit_history"]
    + 1.1 * data["employment_status"]
    - 0.45 * debt_to_income
    - 0.4 * data["existing_loans_count"]
    - 1.2
)
probs = 1.0 / (1.0 + np.exp(-logits))
data["loan_status"] = (probs > 0.50).astype(int)

data.to_csv("data/dataset.csv", index=False)
print("[1/3] Dataset written to data/dataset.csv")

X = data.drop(columns=["loan_status"])
y = data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=42))
])

pipeline.fit(X_train, y_train)
acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"[2/3] Model trained with accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, pipeline.predict(X_test)))

joblib.dump(pipeline, "models/model.pkl")
print("[3/3] Exported pipeline artifact to models/model.pkl")
