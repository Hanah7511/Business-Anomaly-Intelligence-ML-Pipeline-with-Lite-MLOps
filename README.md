# 📊 Business Anomaly Intelligence

### ML Pipeline with Baseline + Isolation Forest + Lite MLOps (CI/CD)

A production-style machine learning pipeline designed to detect anomalies in business metrics such as revenue trends, payment patterns, and transactional behavior using both statistical and ML-based approaches.

This project demonstrates an end-to-end ML pipeline with validation, feature engineering, hybrid anomaly detection, monitoring, and CI-enabled reproducibility — aligned with real-world MLOps practices.

---

## 🚀 Overview

**Business Anomaly Intelligence** is a modular ML pipeline that detects anomalies in key business KPIs including:

* Daily Revenue
* Payment Success & Failure Counts
* Transaction Patterns
* Temporal Business Metrics

The system combines:

* Statistical Baseline Detection (Z-score, IQR, MAD)
* Isolation Forest (ML-based anomaly detection)
* Business Impact Evaluation
* Monitoring & Drift Checks
* CI-enabled automated pipeline execution (Lite MLOps)

This simulates how real companies build anomaly monitoring systems for fintech, e-commerce, and SaaS analytics platforms.

---

## 🏗️ System Architecture

```
Data Source Layer
  ├── SQL Database (Production Mode)
  └── Synthetic Data (CI Mode)
            ↓
Data Validation Layer
  ├── Schema Checks
  ├── Missing Values Detection
  └── Business Logic Validation
            ↓
Feature Engineering Layer
  ├── Time Features
  ├── Rolling Statistics
  ├── Lag & Trend Features
  └── Interaction Features (80+ Features)
            ↓
Hybrid Anomaly Detection
  ├── Baseline Detector (Statistical)
  │     ├── Z-score
  │     ├── IQR
  │     ├── MAD
  │     └── Percentile Methods
  │
  └── ML Detector (Isolation Forest)
        ├── Feature Scaling
        ├── Contamination Control
        └── Anomaly Scoring
            ↓
Evaluation Layer
  ├── Model Agreement Analysis
  ├── Business Impact Evaluation
  └── Alert Rate Monitoring
            ↓
Monitoring Layer
  ├── Data Health Checks
  ├── Drift Detection
  └── Prediction Monitoring
            ↓
Outputs & Artifacts
  ├── Anomaly Flags
  ├── Scores
  ├── Logs
  └── CI Validation Reports
```

---

## 🔄 End-to-End Pipeline Stages

### Stage 1 — Data Extraction

* SQL-based extraction (Production Mode)
* Synthetic data generation (CI Mode)
* Enables testing without database dependency

### Stage 2 — Data Validation

Comprehensive validation including:

* Schema validation
* Missing date detection
* Revenue sanity checks
* Payment consistency validation
* Outlier detection (IQR)
* Business rule validation

Outputs a structured validation report.

### Stage 3 — Feature Engineering

Generates 80+ engineered features:

* Time-based features (day, month, cyclical encoding)
* Rolling statistics
* Lag features
* Trend indicators
* Exponential moving averages
* Interaction features
* Statistical summaries

All missing values handled safely.

### Stage 4 — Baseline Detection (Statistical)

Multi-method anomaly detection:

* Global Z-score
* Rolling Z-score
* IQR Method
* MAD (Median Absolute Deviation)
* Percentile-based detection
* Ensemble anomaly scoring

### Stage 5 — ML Detection

Model: **Isolation Forest**

* Feature scaling
* Contamination-based thresholding
* Anomaly scoring
* Flag generation
* Model persistence support

### Stage 6 — Evaluation Layer

* Baseline vs ML agreement analysis
* Cohen’s Kappa score
* Business impact assessment
* Alert rate health analysis
* Temporal anomaly pattern insights

### Stage 7 — Monitoring Layer

* Data schema checks
* Data freshness monitoring
* Statistical drift detection
* Alert rate monitoring
* Prediction drift checks

Health Status Outputs:

* 🟢 Healthy
* 🟡 Warning
* 🔴 Critical

---

## ⚙️ CI / Lite MLOps Pipeline

This project includes a GitHub Actions CI pipeline.

On every push to `main`:

* Environment setup (Python 3.10)
* Dependency installation
* Synthetic data injection (CI mode)
* Full pipeline execution
* Automated validation checks

### CI ensures:

* Reproducibility
* Automated testing
* Pipeline reliability
* Deployment readiness

This demonstrates foundational **MLOps practices** in a lightweight production setting.

---

## 🧠 Tech Stack

| Category            | Technology                       |
| ------------------- | -------------------------------- |
| Language            | Python 3.10                      |
| Data Processing     | Pandas, NumPy                    |
| ML Model            | Isolation Forest (Scikit-learn)  |
| Feature Engineering | Custom Statistical Features      |
| Validation          | Custom Data Validation Framework |
| Monitoring          | Drift & Health Checks            |
| CI/CD               | GitHub Actions                   |
| Logging             | Python Logging Framework         |

---

## 📊 Sample Output

Pipeline generates:

* `baseline_anomaly_flag`
* `ml_anomaly_flag`
* `ml_anomaly_score`
* Evaluation metrics
* Monitoring reports
* Feature summary (80+ features)

Example:

```
metric_date  daily_revenue  ml_anomaly_flag  ml_anomaly_score
2024-02-12   21515.37       0                -0.088
2024-02-13   20530.18       0                -0.097
2024-02-14   20583.43       0                -0.095
2024-02-15   18765.69       0                -0.061
2024-02-16   22485.45       0                -0.065
```

---

## ▶️ How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python -m src.pipeline.run_pipeline
```

---

## ⭐ Key Highlights

* Modular end-to-end ML pipeline
* Hybrid anomaly detection (Statistical + ML)
* Business-aware evaluation framework
* Monitoring and drift detection included
* CI-enabled reproducibility
* Production-style logging and validation
* Synthetic + Production data compatibility

---

## 🔮 Future Enhancements

* Real-time streaming ingestion (Kafka / APIs)
* Experiment tracking (MLflow)
* Docker containerization
* REST API deployment (FastAPI)
* Automated model retraining
* Monitoring dashboard (Grafana / Streamlit)

---

## 🏷️ Project Classification

**End-to-End Machine Learning Pipeline + Lite MLOps Implementation**

This project covers the full ML lifecycle:
Data → Validation → Features → Model → Evaluation → Monitoring → CI

---

## 👩‍💻 Author

**Hana Al Haris**
AI / ML Engineering Student
Portfolio Project – ML Pipeline & MLOps Systems
