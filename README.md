# Customer Churn Prediction & Retention Intelligence System

An end-to-end, decoupled Machine Learning application built with **Scikit-Learn**, **FastAPI**, and **Streamlit** to predict customer churn risk, identify exact feature-level churn drivers, and support retention decision-making.

---

## System Architecture

The application separates model inference and data validation from the user interface:

```text
customer_churn_system/
├── artifacts/
│   ├── churn_prediction_pipeline.pkl   # Serialized ColumnTransformer + LogisticRegression pipeline
│   └── metrics.json                    # Cross-validation and test evaluation metrics
├── backend/
│   ├── schemas.py                      # Pydantic v2 input/output contracts & batch warning models
│   ├── predictor.py                    # Artifact loading, single/batch scoring, and coefficient-based explainability
│   └── main.py                         # FastAPI REST endpoints (/health, /model-info, /predict, /predict/batch, /predict/csv)
├── frontend/
│   ├── api_client.py                   # Centralized HTTP client connecting Streamlit to FastAPI
│   ├── app.py                          # Multi-page navigation shell and session state manager
│   └── pages/
│       ├── 1_executive_dashboard.py    # Scored population KPIs, risk tier breakdown, and MRR exposure
│       ├── 2_single_prediction.py      # Real-time single customer scoring and top 3 churn reasons
│       ├── 3_batch_prediction.py       # Crash-proof CSV batch scoring with row-level validation warnings
│       ├── 4_high_risk_explorer.py     # Threshold-filtered retention call list and CSV export
│       ├── 5_model_insights.py         # Test metrics and global Logistic Regression feature weights
│       └── 6_prediction_history.py     # Searchable session audit log of all scored predictions
├── notebooks/                          # Data auditing, EDA, preprocessing, and model selection notebooks
├── Dockerfile                          # Multi-service container configuration (FastAPI + Streamlit)
└── requirements.txt                    # Pinned Python dependencies
```

---

## Machine Learning Pipeline & Performance

- **Dataset:** Telco Customer Churn (`tenure`, `MonthlyCharges`, `TotalCharges`, `gender`, `Dependents`, `Contract`)
- **Preprocessing (`ColumnTransformer`):**
  - **Numerical (`tenure`, `MonthlyCharges`, `TotalCharges`):** Median imputation + `RobustScaler`
  - **Nominal Categorical (`gender`, `Dependents`):** Most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
  - **Ordinal Categorical (`Contract`):** Most-frequent imputation + `OrdinalEncoder` (`Month-to-month` < `One year` < `Two year`)
- **Winning Model:** `LogisticRegression(C=10.0, class_weight="balanced", max_iter=1000)` tuned via `GridSearchCV` (`scoring="f1"`)
- **Test Performance:**
  - **Recall:** `0.82` (prioritized to capture at-risk customers early)
  - **F1 Score:** `0.61` (`0.6097` CV F1)
  - **Precision:** `0.48`
  - **Accuracy:** `0.72`
- **Lightweight Explainability:** Computes exact local feature contributions (`X_transformed * coef_`) directly from the linear model weights without external heavy dependencies.

---

## Key Engineering Features

- **Crash-Proof Batch Scoring:** `/predict/batch` and `/predict/csv` validate rows individually, automatically skipping duplicate `customer_id`s, empty rows, or invalid values into structured `RowWarning` reports while scoring all valid rows.
- **Strict API Contracts:** Pydantic v2 schemas enforce value bounds (`ge=0`) and categorical allow-lists (`Literal`) before data reaches the ML pipeline.
- **Stateless Session Analytics:** Uses Streamlit `st.session_state` to power live executive dashboards, high-risk customer filtering, and prediction history with zero external database overhead.

---

## Quickstart (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com
cd Customer-Churn-Prediction-System-Version-02
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend (Terminal 1)
```bash
uvicorn backend.main:app --reload
```
Interactive API documentation will be available at `http://127.0.0`.

### 4. Start the Streamlit Frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.

---

## Run with Docker

```bash
docker build -t churn-intelligence .
docker run -p 7860:7860 churn-intelligence
```
