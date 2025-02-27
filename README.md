# fraud-detection-project
Insurance fraud is a major concern for companies, leading to financial losses and increased claim processing time. This project aims to develop a fraud detection model that can identify fraudulent vehicle insurance claims using machine learning techniques
✅ Detect fraudulent claims based on accident details, vehicle attributes, and policyholder information.
✅ Reduce financial losses by flagging potential fraud cases early.
✅ Improve claim processing efficiency by automating fraud detection
The dataset contains vehicle accident records, policyholder information, and claim details.
Target Variable: FraudFound_P (0 = Non-Fraud, 1 = Fraud)
🔹 Key Features Used
🔹 Vehicle Information: Make, VehiclePrice, AgeOfVehicle, VehicleCategory
🔹 Policy Details: PolicyType, BasePolicy, Days_Policy_Accident, Days_Policy_Claim
🔹 Accident Details: AccidentArea, WitnessPresent, PoliceReportFiled
🔹 Personal Details: AgeOfPolicyHolder, Sex, MaritalStatus, Fault
 Machine Learning Approach
1️⃣ Data Preprocessing:

Handled missing values.
Performed One-Hot Encoding & Label Encoding for categorical features.
Applied Variance Threshold to remove low-variance features.
Used StandardScaler for feature scaling.
2️⃣ Model Training & Evaluation:

Logistic Regression → Baseline model.
Decision Tree & Random Forest → Better fraud detection.
XGBoost → Improved fraud detection using boosting.
Hybrid Model (XGBoost + Random Forest) → Best balance between precision & recall.
3️⃣ Final Model Performance:

Accuracy: 72.19%
Fraud Recall: 82.58% (Detects most fraud cases)
Fraud Precision: 15.38% (Some false positives)
Trade-off: More fraud cases detected but some non-fraud cases flagged incorrectly.
