import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("../data/processed/spacex_enriched.csv")

# -----------------------------
# 2. Feature Selection + Cleaning
# -----------------------------
df['orbit'] = df['orbit_y'].combine_first(df['orbit_x'])
df['launch_site'] = df['launch_site_x'].combine_first(df['launch_site_y'])

df = df[['rocket_name', 'payload_mass', 'orbit', 'landing_success']].dropna()

# Fix types
df['landing_success'] = df['landing_success'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})

# -----------------------------
# 3. Prepare Features and Target
# -----------------------------
X = df[['rocket_name', 'payload_mass', 'orbit']]
y = df['landing_success']

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Preprocessing Pipeline
# -----------------------------
numeric_features = ['payload_mass']
categorical_features = ['rocket_name', 'orbit']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# -----------------------------
# 6. Model Pipeline
# -----------------------------
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# -----------------------------
# 7. Train
# -----------------------------
clf.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate
# -----------------------------
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 9. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

