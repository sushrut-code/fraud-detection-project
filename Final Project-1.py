
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn import set_config
from sklearn.metrics import roc_curve, auc

set_config(display='diagram')
np.random.seed(42)

# 📥 2. Load and Preprocess Data
df = pd.read_csv("C:\\Users\\LENOVO\\Documents\\fraud_oracle_prac.csv")
df = df.drop(['PolicyNumber', 'RepNumber'], axis=1)

# Clean WeekOfMonth and WeekOfMonthClaimed
week_mapping = {1: "week-1", 2: "week-2", 3: "week-3", 4: "week-4", 5: "week-5"}
df['WeekOfMonth'] = df['WeekOfMonth'].map(week_mapping)
df['WeekOfMonthClaimed'] = df['WeekOfMonthClaimed'].map(week_mapping)

# Group rare 'Make' values
make_counts = df['Make'].value_counts()
rare_makes = make_counts[make_counts <= 100].index
df['Make'] = df['Make'].replace(rare_makes, 'uncommon')

# 🧪 3. Split Data
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧪 4. Handle Class Imbalance (Upsample Fraud)
train_data = pd.concat([X_train, y_train], axis=1)
fraud, non_fraud = train_data[train_data.FraudFound_P == 1], train_data[train_data.FraudFound_P == 0]
fraud_upsampled = resample(fraud, replace=True, n_samples=len(non_fraud), random_state=42)
balanced_train = pd.concat([non_fraud, fraud_upsampled])
X_train = balanced_train.drop('FraudFound_P', axis=1)
y_train = balanced_train['FraudFound_P']

# 🏷️ 5. Preprocessing
categorical_cols = [
    'Month', 'WeekOfMonth', 'DayOfWeek', 'AccidentArea', 'DayOfWeekClaimed',
    'MonthClaimed', 'WeekOfMonthClaimed', 'Sex', 'MaritalStatus', 'Fault',
    'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Deductible', 'DriverRating',
    'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims',
    'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent',
    'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars',
    'Year', 'BasePolicy', 'Make'
]
log_transform_cols = ['Age']

preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('log_age', FunctionTransformer(np.log1p, validate=False), log_transform_cols)
], remainder='passthrough')

# 🔁 6. Model Training Function
def train_and_evaluate(model, name):
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n📊 {name} Performance")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return pipeline

# ⚙️ 7. Train Models
log_reg = LogisticRegression(class_weight='balanced', solver='liblinear')
decision_tree = DecisionTreeClassifier(max_depth=20)
random_forest = RandomForestClassifier(n_estimators=100)
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

pipe_lr = train_and_evaluate(log_reg, "Logistic Regression")
pipe_dt = train_and_evaluate(decision_tree, "Decision Tree")
pipe_rf = train_and_evaluate(random_forest, "Random Forest")
pipe_xgb = train_and_evaluate(xgboost, "XGBoost")

# 🧠 8. Hybrid Model (Stacked Ensemble)
rf_probs = pipe_rf.predict_proba(X_test)[:, 1]
xgb_probs = pipe_xgb.predict_proba(X_test)[:, 1]
stacked_features = np.column_stack((xgb_probs, rf_probs))

meta_model = LogisticRegression(class_weight='balanced')
meta_model.fit(stacked_features, y_test)
hybrid_preds = meta_model.predict(stacked_features)

print("\n🤖 Hybrid Model (XGBoost + Random Forest)")
print("Accuracy:", accuracy_score(y_test, hybrid_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, hybrid_preds))
print("Classification Report:\n", classification_report(y_test, hybrid_preds))

# Get prediction probabilities
lr_probs = pipe_lr.predict_proba(X_test)[:, 1]
dt_probs = pipe_dt.predict_proba(X_test)[:, 1]
rf_probs = pipe_rf.predict_proba(X_test)[:, 1]
xgb_probs = pipe_xgb.predict_proba(X_test)[:, 1]
hybrid_probs = meta_model.predict_proba(np.column_stack((xgb_probs, rf_probs)))[:, 1]

# Compute ROC and AUC
def get_roc(name, probs):
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, name

models_roc = [
    get_roc("Logistic Regression", lr_probs),
    get_roc("Decision Tree", dt_probs),
    get_roc("Random Forest", rf_probs),
    get_roc("XGBoost", xgb_probs),
    get_roc("Hybrid Model", hybrid_probs),
]

# Plotting
plt.figure(figsize=(10, 6))
for fpr, tpr, roc_auc, label in models_roc:
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guess
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

import pickle

# Pack everything needed to reload and use the hybrid model
hybrid_bundle = {
    "pipe_rf": pipe_rf,
    "pipe_xgb": pipe_xgb,
    "meta_model": meta_model,
    "preprocessor": preprocessor,  # optional but useful if needed
    "feature_names": X_train.columns.tolist()  # for later reference
}

# Save using pickle
with open("hybrid_fraud_model.pkl", "wb") as f:
    pickle.dump(hybrid_bundle, f)

print("✅ Hybrid model and pipelines successfully saved to 'hybrid_fraud_model.pkl'")


import pickle
import numpy as np

# Load the model
with open("hybrid_fraud_model.pkl", "rb") as f:
    hybrid_bundle = pickle.load(f)

# Access components
pipe_rf_loaded = hybrid_bundle["pipe_rf"]
pipe_xgb_loaded = hybrid_bundle["pipe_xgb"]
meta_model_loaded = hybrid_bundle["meta_model"]

# Example: Make predictions on new data `X_new`
rf_probs = pipe_rf_loaded.predict_proba(X_new)[:, 1]
xgb_probs = pipe_xgb_loaded.predict_proba(X_new)[:, 1]
stacked_input = np.column_stack((xgb_probs, rf_probs))
hybrid_predictions = meta_model_loaded.predict(stacked_input)
