import pandas as pd

# ---------------- DIABETES ----------------
diabetes = pd.read_csv("diabetes.csv")

# Rename columns safely
diabetes.columns = diabetes.columns.str.strip()

diabetes = diabetes.rename(columns={
    'Glucose': 'glucose',
    'BMI': 'bmi',
    'Age': 'age',
    'BloodPressure': 'bp',
    'Outcome': 'diabetes'
})

# Add missing columns
diabetes['smoking'] = 0
diabetes['exercise'] = 1
diabetes['family'] = 0

# Other diseases
diabetes['heart'] = 0
diabetes['obesity'] = (diabetes['bmi'] > 30).astype(int)


# ---------------- HEART ----------------
heart = pd.read_csv("heart.csv")
heart.columns = heart.columns.str.strip()

print("Heart Columns:", heart.columns)  # DEBUG

# Rename safely
if 'trestbps' in heart.columns:
    heart.rename(columns={'trestbps': 'bp'}, inplace=True)
elif 'RestingBP' in heart.columns:
    heart.rename(columns={'RestingBP': 'bp'}, inplace=True)

if 'chol' in heart.columns:
    heart.rename(columns={'chol': 'glucose'}, inplace=True)
elif 'Cholesterol' in heart.columns:
    heart.rename(columns={'Cholesterol': 'glucose'}, inplace=True)

if 'target' in heart.columns:
    heart.rename(columns={'target': 'heart'}, inplace=True)
elif 'HeartDisease' in heart.columns:
    heart.rename(columns={'HeartDisease': 'heart'}, inplace=True)

# Add missing if still not present
if 'glucose' not in heart.columns:
    heart['glucose'] = 120

if 'bp' not in heart.columns:
    heart['bp'] = 120

if 'heart' not in heart.columns:
    heart['heart'] = 0

# Add other required columns
heart['bmi'] = 25
heart['smoking'] = 0
heart['exercise'] = 1
heart['family'] = 0
heart['diabetes'] = 0
heart['obesity'] = 0

# ---------------- OBESITY ----------------
obesity = pd.read_csv("obesity.csv")

obesity.columns = obesity.columns.str.strip()

# Rename basic columns
obesity = obesity.rename(columns={
    'Age': 'age',
    'Height': 'height',
    'Weight': 'weight'
})

# Convert height if in meters
if obesity['height'].max() < 3:
    obesity['height'] = obesity['height'] * 100

# Create BMI
obesity['bmi'] = obesity['weight'] / ((obesity['height']/100) ** 2)

# Detect obesity column automatically
possible_cols = ['Obesity_Level', 'Label', 'NObeyesdad', 'obesity']

found = False
for col in possible_cols:
    if col in obesity.columns:
        obesity['obesity'] = obesity[col].apply(lambda x: 1 if "Obese" in str(x) else 0)
        found = True
        break

# Fallback if no column found
if not found:
    obesity['obesity'] = (obesity['bmi'] > 30).astype(int)

# Add missing columns
obesity['glucose'] = 120
obesity['bp'] = 120
obesity['smoking'] = 0
obesity['exercise'] = 1
obesity['family'] = 0

# Other diseases
obesity['diabetes'] = 0
obesity['heart'] = 0

# ---------------- FINAL MERGE ----------------
final_df = pd.concat([
    diabetes[['age','bmi','glucose','bp','smoking','exercise','family','diabetes','heart','obesity']],
    heart[['age','bmi','glucose','bp','smoking','exercise','family','diabetes','heart','obesity']],
    obesity[['age','bmi','glucose','bp','smoking','exercise','family','diabetes','heart','obesity']]
])

# Save final dataset
final_df.to_csv("final_health_data.csv", index=False)

print("✅ Data prepared and merged successfully!")
print("Shape:", final_df.shape)