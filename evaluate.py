import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Simulate some sensor data
np.random.seed(42)
data_size = 1000
data = {
    'temperature': np.random.normal(loc=75, scale=5, size=data_size),  # Simulated temperature data
    'vibration': np.random.normal(loc=0.5, scale=0.1, size=data_size),  # Simulated vibration data
    'current': np.random.normal(loc=10, scale=1, size=data_size),      # Simulated electrical current data
    'failure': np.random.choice([0, 1], size=data_size, p=[0.9, 0.1])  # Simulated failure labels (10% failure rate)
}

df = pd.DataFrame(data)

# Feature engineering: let's assume high temperature and vibration indicate failure
df['temp_vib_ratio'] = df['temperature'] / df['vibration']

# Split the data into training and testing sets
X = df[['temperature', 'vibration', 'current', 'temp_vib_ratio']]
y = df['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices])
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()
