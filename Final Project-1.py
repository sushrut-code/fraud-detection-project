import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Load Dataset
df = pd.read_csv("C:\\Users\\LENOVO\\Documents\\fraud_oracle_prac.csv")

# One-hot encode selected categorical columns
categorical_cols = [
    'Month', 'DayOfWeek', 'Make', 'DayOfWeekClaimed', 'MonthClaimed',
    'MaritalStatus', 'PolicyType', 'VehicleCategory', 'VehiclePrice',
    'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims',
    'AgeOfVehicle', 'AgeOfPolicyHolder', 'NumberOfSuppliments',
    'AddressChange_Claim', 'NumberOfCars', 'BasePolicy'
]
df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# Binary encoding for specific features
binary_map = {
    'AccidentArea': {'Urban': 0, 'Rural': 1},
    'Sex': {'Male': 0, 'Female': 1},
    'Fault': {'Policy Holder': 0, 'Third Party': 1},
    'PoliceReportFiled': {'No': 0, 'Yes': 1},
    'WitnessPresent': {'No': 0, 'Yes': 1},
    'AgentType': {'External': 0, 'Internal': 1}
}
df.replace(binary_map, inplace=True)

# Drop unnecessary columns
df.drop(['PolicyNumber', 'RepNumber'], axis=1, inplace=True)

# Features and target
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Upsample minority class in training set
train_data = pd.concat([X_train, y_train], axis=1)
fraud = train_data[train_data['FraudFound_P'] == 1]
non_fraud = train_data[train_data['FraudFound_P'] == 0]
fraud_upsampled = resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42)
train_balanced = pd.concat([non_fraud, fraud_upsampled])

X_train = train_balanced.drop('FraudFound_P', axis=1)
y_train = train_balanced['FraudFound_P']

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Hybrid Model: XGBoost + Random Forest
xgb_probs = models["XGBoost"].predict_proba(X_test_scaled)[:, 1]
rf_probs = models["Random Forest"].predict_proba(X_test_scaled)[:, 1]
stacked_input = np.column_stack((xgb_probs, rf_probs))

meta_model = LogisticRegression(class_weight='balanced')
meta_model.fit(stacked_input, y_test)
hybrid_preds = meta_model.predict(stacked_input)

print("\nHybrid Model (XGBoost + Random Forest)")
print("Accuracy:", accuracy_score(y_test, hybrid_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, hybrid_preds))
print("Classification Report:\n", classification_report(y_test, hybrid_preds))

# Plot ROC Curve for XGBoost and Random Forest
plt.figure(figsize=(10, 6))

for name in ["Random Forest", "XGBoost"]:
    y_scores = models[name].predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Hybrid ROC
fpr, tpr, _ = roc_curve(y_test, meta_model.predict_proba(stacked_input)[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"Hybrid Model (AUC = {roc_auc:.2f})", linestyle='--')

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
