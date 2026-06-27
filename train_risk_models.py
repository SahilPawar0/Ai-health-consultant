import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

print("Generating dataset and training models...")

# 1. Generate Synthetic Patient Data (1000 rows)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'weight': np.random.uniform(50, 120, n_samples),
    'height': np.random.uniform(150, 190, n_samples),
    'glucose': np.random.uniform(70, 200, n_samples), # Blood sugar
    'bp': np.random.uniform(90, 180, n_samples),     # Blood pressure
    'family': np.random.choice([0, 1], n_samples),   # 0=No, 1=Yes
    'gender': np.random.choice([0, 1], n_samples),   # 0=Female, 1=Male
    'smoke': np.random.choice([0, 1], n_samples),    # 0=No, 1=Yes
    'active': np.random.choice([1, 2, 3], n_samples) # 1=Low, 2=Mod, 3=High
})

# Calculate BMI
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

# Define target variables based on medical logic for realistic training
data['diabetes_target'] = ((data['glucose'] > 140) & (data['bmi'] > 25)).astype(int)
data['heart_target'] = ((data['bp'] > 130) & (data['age'] > 50)).astype(int)
data['obesity_target'] = (data['bmi'] > 30).astype(int)

# ---------------------------------------------------------
# MODEL 1: Diabetes (Needs: age, bmi, glucose, bp, family)
# ---------------------------------------------------------
X_diabetes = data[['age', 'bmi', 'glucose', 'bp', 'family']]
y_diabetes = data['diabetes_target']

diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_diabetes, y_diabetes)
joblib.dump(diabetes_model, "model/diabetes_model.pkl")
print("✅ Diabetes model trained and saved!")

# ---------------------------------------------------------
# MODEL 2: Heart Disease (Needs: age, bp, glucose)
# ---------------------------------------------------------
X_heart = data[['age', 'bp', 'glucose']]
y_heart = data['heart_target']

heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
heart_model.fit(X_heart, y_heart)
joblib.dump(heart_model, "model/heart_model.pkl")
print("✅ Heart Disease model trained and saved!")

# ---------------------------------------------------------
# MODEL 3: Obesity (Needs: age, bmi, gender, smoke, active)
# ---------------------------------------------------------
X_obesity = data[['age', 'bmi', 'gender', 'smoke', 'active']]
y_obesity = data['obesity_target']

obesity_model = RandomForestClassifier(n_estimators=100, random_state=42)
obesity_model.fit(X_obesity, y_obesity)
joblib.dump(obesity_model, "model/obesity_model.pkl")
print("✅ Obesity model trained and saved!")

print("\nAll models successfully updated! You can now use the Risk Predictor.")