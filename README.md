# E-commerce Real-Time ML Pipeline on Azure

This project implements an **end-to-end real-time machine learning system** for an e-commerce use case, covering data ingestion, analytics, feature engineering, model training, CI/CD, serving, and monitoring.

The focus is not just model accuracy, but **production-grade ML systems** — how data flows, how models are trained safely, deployed, monitored, and retrained.

---

## 🧩 Architecture Overview

**Data Flow**
1. Web/App events → Azure Event Hubs  
2. Stream Analytics → ADLS Gen2 (Bronze)  
3. Databricks → Silver (cleaned, deduplicated)  
4. Databricks → Gold (analytics + feature tables)  

**ML Flow**
1. Feature & label generation (Gold layer)
2. Python training pipeline (scikit-learn)
3. CI/CD with GitHub Actions
4. Real-time inference via FastAPI
5. Drift monitoring & retraining triggers

---

## 🥉 Bronze Layer (Raw Ingestion)

- Real-time event ingestion using **Azure Event Hubs**
- Stream Analytics writes raw JSON events into **ADLS Gen2**
- No transformations, schema preserved
- Purpose: auditability and replay

---

## 🥈 Silver Layer (Clean & Standardized)

Implemented in Databricks:
- Schema enforcement
- Event deduplication (`event_id`)
- Timestamp normalization
- Data quality checks
- Session and user alignment

Output: **trusted, analytics-ready event tables**

---

## 🥇 Gold Layer (Analytics & Features)

### Analytics tables
- `fact_events`
- `dim_customer`

### Feature tables
- Daily customer behavioral features (sessions, views, carts, purchases)
- Designed for **reproducibility and ML training**

### Labels
- `label_purchase_next_7d`
- Supports purchase propensity modeling

---

## 🤖 Model Training

- Python pipeline using `pandas` + `scikit-learn`
- User-based split (fallback when time-based split not possible)
- Metrics:
  - ROC-AUC
  - PR-AUC
- Artifacts:
  - `propensity_model.joblib`
  - `metrics.json`

---

## 🔁 CI/CD (MLOps-lite)

GitHub Actions workflows:
- Linting & unit tests on every PR
- Smoke training job on `main`
- Metric-based promotion gate
- Candidate model artifacts stored automatically

---

## 🚀 Serving

### Real-time inference
- FastAPI service
- `/health` and `/score` endpoints
- Model loaded from trained artifact
- Designed to be container-ready (Docker skipped locally due to OS constraints)

### Batch scoring
- Offline scoring for campaigns and CRM activation
- Outputs customer-level propensity scores

---

## 📊 Monitoring & Retraining

- Feature drift detection using **Population Stability Index (PSI)**
- Prediction drift monitoring
- Automated retraining trigger logic
- Daily drift check workflow via GitHub Actions

---

## 🧠 Key Takeaways

- ML systems are **more than models**
- Real impact comes from:
  - reliable data pipelines
  - safe training
  - deployment discipline
  - monitoring & feedback loops

---

## 🛠️ Tech Stack

- Azure Event Hubs
- Azure Stream Analytics
- Azure Data Lake Gen2
- Databricks
- Python (pandas, scikit-learn)
- FastAPI
- GitHub Actions

---

## 📌 Status

This project is designed as a **production-style reference architecture** and can be extended with:
- Azure ML Managed Online Endpoints
- Time-based retraining
- Feature Store integration
- Online experimentation (A/B testing)

---

