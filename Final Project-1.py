'''Vehicle insurance fraud, such as staged accidents and exaggerated claims, leads to significant financial losses for insurers.
 Using a dataset with vehicle attributes, accident details, and policy information, the objective is to develop a fraud detection model
   to identify fraudulent claims, uncover key risk factors, and enhance prevention strategies to reduce fraud and streamline claim processing.'''




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

# Load the dataset
data = pd.read_csv("fraud_oracle.csv")

# Display basic information about the dataset
print(data.head())
print(data.info())

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Check for duplicate rows
duplicated_values = data.duplicated().sum()
print("Duplicate values:", duplicated_values)

# Check unique values in 'Age' column
unique_values = data['Age'].unique()
print("Unique values in Age column:", unique_values)

# Check fraud distribution
fraud_distribution = data['FraudFound_P'].value_counts(normalize=True) * 100
print("Fraud Distribution:\n", fraud_distribution)

# Visualizing fraud distribution
plt.figure(figsize=(8, 8))
fraud_distribution.plot(kind='pie', autopct="%1.2f%%", labels=["Non-Fraud", "Fraud"], colors=['lightblue', 'red'])
plt.title("Distribution of Fraud vs Non-Fraud")
plt.ylabel("")
plt.show()

# Encoding categorical variables
binary_cols = ['AccidentArea', 'Sex', 'Fault', 'PoliceReportFiled', 'WitnessPresent', 'AgentType']
encoder = LabelEncoder()
for col in binary_cols:
    data[col] = encoder.fit_transform(data[col])

# One-hot encoding for categorical features
one_hot_cols = ['Month', 'DayOfWeek', 'MonthClaimed', 'DayOfWeekClaimed', 'Make', 'PolicyType',
                'VehicleCategory', 'VehiclePrice', 'BasePolicy', 'MaritalStatus', 'Days_Policy_Accident',
                'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder',
                'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars']

data = pd.get_dummies(data, columns=one_hot_cols, drop_first=True)

# Drop irrelevant columns
data.drop(['PolicyNumber', 'RepNumber'], axis=1, inplace=True)

# Feature Selection: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
reduced_data = selector.fit_transform(data)
selected_features = data.columns[selector.get_support()]
reduced_data = pd.DataFrame(reduced_data, columns=selected_features)

# Split dataset into features and target variable
X = reduced_data.drop(columns=['FraudFound_P'])
y = reduced_data['FraudFound_P']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]), n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Hybrid Model: Combining XGBoost and Random Forest
xgb_preds = models['XGBoost'].predict_proba(X_test)[:, 1]
rf_preds = models['Random Forest'].predict_proba(X_test)[:, 1]
stacked_features = np.column_stack((xgb_preds, rf_preds))

# Meta-classifier
meta_classifier = LogisticRegression(class_weight='balanced')
meta_classifier.fit(stacked_features, y_test)
final_preds = meta_classifier.predict(stacked_features)

# Evaluate Hybrid Model
print("\nHybrid Model (XGBoost + Random Forest) Results:")
print(f"Accuracy: {accuracy_score(y_test, final_preds):.4f}")
print(f"Precision: {precision_score(y_test, final_preds):.4f}")
print(f"Recall: {recall_score(y_test, final_preds):.4f}")
print(f"F1-score: {f1_score(y_test, final_preds):.4f}")
print("\nClassification Report:\n", classification_report(y_test, final_preds))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_preds))

# Save the trained hybrid model
joblib.dump(meta_classifier, "hybrid_fraud_detection.pkl")
print("Model saved successfully!")

'''Final Model Summary
 Detects 82.58% of fraud cases (High Recall) 
 Good overall fraud detection balance (F1-score: 25.93%) 
 Acceptable accuracy (72.19%) 
 Trade-off: More fraud cases detected, but some false positives remain.'''
