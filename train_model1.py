import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ================= LOAD & PREPARE DATA =================

# ---------- DIABETES ----------
diabetes = pd.read_csv("dataset/diabetes.csv")
diabetes.columns = diabetes.columns.str.strip()

# Rename columns if needed
diabetes = diabetes.rename(columns={
    'Pregnancies':'pregnancies',
    'Glucose':'glucose',
    'BloodPressure':'bp',
    'SkinThickness':'skin',
    'Insulin':'insulin',
    'BMI':'bmi',
    'DiabetesPedigreeFunction':'family',
    'Age':'age',
    'Outcome':'diabetes'
})

# Features & target
X_d = diabetes[['age','bmi','glucose','bp','family']]
y_d = diabetes['diabetes']

# Split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, shuffle=True
)

# Model
diabetes_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
diabetes_model.fit(X_train_d, y_train_d)
d_pred = diabetes_model.predict(X_test_d)
print("Diabetes Accuracy:", round(accuracy_score(y_test_d, d_pred), 3))

# ---------- HEART ----------
heart = pd.read_csv("dataset/heart.csv")
heart.columns = heart.columns.str.strip()

# Features: age, average BP, cholesterol as glucose proxy
heart['bp'] = (heart['ap_hi'] + heart['ap_lo']) / 2
heart['glucose'] = heart['cholesterol']  # proxy
heart['heart'] = heart['cardio']

X_h = heart[['age','bp','glucose']]
y_h = heart['heart']

# Split
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42, shuffle=True
)

heart_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
heart_model.fit(X_train_h, y_train_h)
h_pred = heart_model.predict(X_test_h)
print("Heart Accuracy:", round(accuracy_score(y_test_h, h_pred), 3))

# ---------- OBESITY ----------
obesity = pd.read_csv("dataset/obesity.csv")
obesity.columns = obesity.columns.str.strip()

# Rename & BMI
obesity = obesity.rename(columns={
    'Age':'age','Height':'height','Weight':'weight','Obesity':'obesity'
})
if obesity['height'].max() < 3:
    obesity['height'] = obesity['height']*100
obesity['bmi'] = obesity['weight']/((obesity['height']/100)**2)

# Map categorical columns to numeric
obesity['gender'] = obesity['Gender'].map({'Male':1,'Female':0})
obesity['smoke'] = obesity['FAVC'].map({'yes':1,'no':0})      # lifestyle proxy
obesity['active'] = obesity['FAF']                             # activity hours

# Target
obesity['obesity'] = obesity['obesity'].apply(lambda x: 1 if "Obesity" in str(x) else 0)

# Features
feature_cols = ['age','bmi','gender','smoke','active']
X_o = obesity[feature_cols]
y_o = obesity['obesity']

# Split
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_o, y_o, test_size=0.2, random_state=42, shuffle=True
)

# Model: reduced depth to avoid 1.0 accuracy
obesity_model = RandomForestClassifier(
    n_estimators=100, max_depth=3, random_state=42
)
obesity_model.fit(X_train_o, y_train_o)
o_pred = obesity_model.predict(X_test_o)
print("Obesity Accuracy:", round(accuracy_score(y_test_o, o_pred), 3))

# ================= SAVE MODELS =================
joblib.dump(diabetes_model, "diabetes_model.pkl")
joblib.dump(heart_model, "heart_model.pkl")
joblib.dump(obesity_model, "obesity_model.pkl")

print("\n✅ All models trained separately and saved!")