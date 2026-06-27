import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# load dataset
df = pd.read_csv("dataset/Training.csv")

# remove prognosis column
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# train model
model = RandomForestClassifier()
model.fit(X, y)

# save model
pickle.dump(model, open("model/disease_model.pkl","wb"))

# save symptom list
pickle.dump(list(X.columns), open("model/symptoms.pkl","wb"))

print("Model trained successfully")
