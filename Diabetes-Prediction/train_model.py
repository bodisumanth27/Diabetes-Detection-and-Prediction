import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset URL
url = (
    "https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

# Column names
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Load dataset
data = pd.read_csv(url, header=None, names=columns)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)
print(data.head())

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Treat physiologically implausible zero measurements
# as missing for these columns.
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, float("nan"))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build preprocessing and ML pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on held-out data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel training completed!")
print(f"Test accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the complete pipeline
joblib.dump(model, "diabetes_model.pkl")

print("\nModel saved as diabetes_model.pkl")