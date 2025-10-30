# 🧠 Loan Default Prediction using Machine Learning

---

## 🎯 Objective
Build a machine learning model to predict whether a loan applicant will default or not based on their financial and demographic information.

---

## 📦 Dataset
**Source:** [Loan Default Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

---

## 📊 Dataset Overview
- **Records:** ~148,000  
- **Target Variable:** `Status` → (0 = No Default, 1 = Default)  
- **Features:** Applicant information, credit score, income, loan details, and more.

---

## 🧰 Libraries Used
- pandas  
- numpy  
- scikit-learn  
- lightgbm  
- joblib  
- streamlit  

---

## ⚙️ Project Workflow

### 1️⃣ Data Splitting
The dataset was split into **training** and **testing** sets using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X = df.drop("Status", axis=1)
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 2️⃣ Preprocessing
Separate pipelines were created for **numerical** and **categorical** features using `ColumnTransformer`.

```python




num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


```

---

### 3️⃣ Model Building — LightGBM Classifier
Used **LightGBM**, a fast and efficient boosting algorithm, within a pipeline.

```python

lgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier(random_state=42))
])
```

---

### 4️⃣ Hyperparameter Tuning
Performed randomized search for optimized parameters.

```python


param_dist = {
    'model__n_estimators': [100, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [-1, 5, 7, 9],
    'model__num_leaves': [15, 31, 63],
    'model__subsample': [0.7, 0.8, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
}

```

---

### 5️⃣ Model Evaluation
Evaluated using key performance metrics.

```python

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

---

## 📈 Key Metrics

| Metric | Score |
|:--------|:-------:|
| Accuracy | 0.87 |
| Precision | 0.94 |
| Recall | 0.47 |
| F1-score | 0.63 |
| ROC-AUC | 0.83 |

---

### 6️⃣ Model Saving
Saved the best pipeline model for deployment.

```python
import joblib
joblib.dump(search.best_estimator_, 'loan_default_pipeline.pkl')
```

✅ **Model saved as:** `loan_default_pipeline.pkl`

---

## 💻 Streamlit App Integration
A simple **Streamlit interface** was built to make real-time predictions using the saved model.

```python


@st.cache_resource
def load_model():
    return joblib.load("loan_default_pipeline.pkl")

pipeline = load_model()

st.title("🏦 Loan Default Prediction App")

loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=1000000, step=1000)
income = st.number_input("Applicant Income", min_value=0, max_value=1000000, step=1000)
gender = st.selectbox("Gender", ["Male", "Female"])


```

---

## 🚀 Next Steps
- Add more ML models (**Decision Tree**, **Random Forest**, **XGBoost**)  
- Perform deeper hyperparameter tuning (**GridSearchCV**)  
- Deploy model using **Flask / FastAPI**  
- Add explainability tools (**SHAP / LIME**)  
- Enhance UI for user-friendly prediction input  

---

## 👨‍💻 Author
**Revan Mahesh Gaikwad**  
*Machine Learning & Data Science Enthusiast*  

