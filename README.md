# 🔮 Customer Churn Prediction System

A production-ready end-to-end Machine Learning system that predicts customer churn for telecom companies using XGBoost, achieving **AUC of 0.9364** — exceeding industry benchmark of 0.85.

---

## 📸 Demo

> Single Prediction | Bulk Upload | What-If Analysis | Model Comparison

---

## 🧠 What Problem Does This Solve?

Telecom companies lose millions every month to customer churn. Acquiring a new customer costs **5-7x more** than retaining one. This system identifies which customers are likely to leave — before they do — so retention teams can intervene with targeted offers.

---

## ⚙️ System Architecture
Raw Data → Feature Engineering → SMOTE Balancing → XGBoost (Optuna Tuned)
↓
FastAPI REST API
↓
Streamlit Dashboard (5 pages)
---

## 📊 Model Performance

| Metric    | Score  | Industry Standard |
|-----------|--------|-------------------|
| AUC       | 0.9364 | 0.75 - 0.85       |
| Precision | 0.7949 | 0.65 - 0.75       |
| Recall    | 0.8464 | 0.60 - 0.75       |
| F1 Score  | 0.8198 | 0.65 - 0.75       |

---

## 🚀 Features

- **Single Prediction** — predict churn for one customer with SHAP explainability
- **Bulk CSV Upload** — score thousands of customers at once, download results
- **What-If Analysis** — simulate retention scenarios (discount, contract upgrade, etc.)
- **Analytics Dashboard** — BI overview with churn by state, contract type, tenure
- **Model Comparison** — XGBoost vs Random Forest vs Gradient Boosting vs Logistic Regression
- **REST API** — FastAPI with Swagger docs and batch prediction endpoint
- **18/18 Tests Passing** — full test coverage across data, model, and API layers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, scikit-learn, SHAP |
| Hyperparameter Tuning | Optuna (30 trials) |
| Class Imbalance | SMOTE (imbalanced-learn) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |

---

## 📁 Project Structure
churn-prediction/
├── src/
│   ├── data/          # Data generation, loading, validation
│   ├── features/      # Feature engineering & preprocessing pipeline
│   ├── models/        # Training, evaluation, prediction, SHAP explainer
│   ├── api/           # FastAPI routes, schemas, main app
│   └── dashboard/     # Streamlit components and charts
├── pages/             # Multi-page Streamlit app
├── tests/             # Pytest test suite
├── data/              # Raw and processed data
├── models/            # Saved model artifacts
└── logs/              # Application logs and plots


---

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Setup environment
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt --prefer-binary

# Generate data and train model
python -m src.data.generator
python -m src.features.pipeline
python -m src.models.trainer

# Run API (Terminal 1)
python -m uvicorn src.api.main:app --reload --port 8000

# Run Dashboard (Terminal 2)
streamlit run app1.py
```

- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 👤 Author

**Sam** — [GitHub](https://github.com/YOUR_USERNAME)