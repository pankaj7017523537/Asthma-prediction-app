import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("C:/Users/HP/Downloads/Asthma ML model/asthma_disease_data.csv")

# Use only selected features
selected_features = ['Age', 'PollutionExposure', 'PollenExposure', 'SleepQuality',
                     'PhysicalActivity', 'DustExposure', 'DietQuality', 'Diagnosis']
df = df[selected_features]

# Check class distribution
print("Diagnosis\n", df["Diagnosis"].value_counts(normalize=True))

# Split features and labels
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📋 Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, "asthma_model.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")
print("🎉 Model and feature names saved!")

# Test with example input (same 7 features)
example_input = [25, 6, 6, 7, 5, 4, 8]
prediction = model.predict([example_input])
print("🧠 Predicted Diagnosis (0 = No Asthma, 1 = Asthma):", prediction[0])
