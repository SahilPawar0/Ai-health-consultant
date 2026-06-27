import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("dataset/Training.csv")

# Remove empty column if present
if 'Unnamed: 133' in data.columns:
    data = data.drop("Unnamed: 133", axis=1)

# Split features and target
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Train model
model = RandomForestClassifier(n_estimators=200)

model.fit(X, y)

print("Model trained successfully")

# Prediction function
def predict_disease(selected_symptoms):

    input_data = np.zeros(len(X.columns))

    # Convert symptoms into model input
    for symptom in selected_symptoms:
        if symptom in X.columns:
            index = X.columns.get_loc(symptom)
            input_data[index] = 1

    input_data = input_data.reshape(1,-1)

    # Get probability
    probabilities = model.predict_proba(input_data)[0]

    diseases = model.classes_

    results = []

    for disease,prob in zip(diseases, probabilities):

        percent = round(prob*100,2)

        if percent > 30:   # threshold
            results.append((disease, percent))

    # Sort by probability
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Maximum 4 results
    results = results[:4]

    return results


# Example test
symptoms = ["fever","vomiting","fatigue","headache"]

prediction = predict_disease(symptoms)

print("\nPossible Diseases:")

for disease,percent in prediction:
    print(f"{disease} : {percent}%")
