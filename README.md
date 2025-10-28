# 🧠 Loan Default Prediction using Machine Learning

## 🎯 Objective
Build a machine learning model to predict whether a loan applicant will default or not based on their financial and demographic information.

---
Dataset Link : https://www.kaggle.com/datasets/yasserh/loan-default-dataset

---
## 📊 Dataset Overview
- **Records:** ~148,000  
- **Target Variable:** `Status` → (0 = No Default, 1 = Default)  
- **Features:** Applicant information, credit score, income, loan details, etc.

---

## 🧰 Libraries Used
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- joblib  

---

## ⚙️ Steps Followed

### 1️⃣ Data Splitting
The dataset was split into training and testing sets using `train_test_split`.

### 2️⃣ Preprocessing
Created pipelines for both numerical and categorical features.

```
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_col = X_train.select_dtypes(include=['int64','float64']).columns
cat_col = X_train.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_col),
    ('cat', cat_pipeline, cat_col)
])
```
3️⃣ Model Building
Combined preprocessing and model into a single pipeline.

```
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])
```
4️⃣ Model Training

```
model_pipeline.fit(X_train, y_train)
```
5️⃣ Model Evaluation
Evaluated using multiple metrics.

```
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

6️⃣ Model Saving

```
import joblib
joblib.dump(model_pipeline, 'loan_default_model.pkl')
```
📈 Key Metrics

Metric	Score

- Accuracy: 0.8651711845025897

- Precision: 0.9481276005547851

- Recall: 0.47209944751381216

- F1-score: 0.630336560627017

- ROC-AUC: 0.8288783577400873


---

🚀 Next Steps

- Add more models (Decision Tree, Random Forest, XGBoost)


- Perform Hyperparameter Tuning using GridSearchCV


- Deploy model using Flask or FastAPI

---

👨‍💻 Author

Revan Mahesh Gaikwad

(Machine Learning & Data Science Enthusiast)

