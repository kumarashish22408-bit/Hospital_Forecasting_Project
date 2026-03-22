import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("HDHI_Admission_data.csv")
data["D.O.A"] = pd.to_datetime(data["D.O.A"], errors="coerce")
data = data.dropna(subset=["D.O.A"])

# Convert date to ordinal
data["date_ordinal"] = data["D.O.A"].map(lambda x: x.toordinal())

# Daily count
daily_counts = data.groupby("date_ordinal").size().reset_index(name="Patient_Count")

X = daily_counts[["date_ordinal"]]
y = daily_counts["Patient_Count"]

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("rf_model.pkl", "wb"))

print("Random Forest Model Trained Successfully!")