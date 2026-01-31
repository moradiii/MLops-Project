# MLOps Project – Customer Churn Prediction

## Project Description
This project focuses on the implementation of an end-to-end **MLOps pipeline** for a customer churn prediction use case.

The goal is to apply MLOps principles across the entire machine learning lifecycle, including data handling, model training, experiment tracking, reproducibility, model serving, and monitoring.  
The emphasis of this project is on **machine learning operations (MLOps)** rather than traditional DevOps practices.

The project is developed incrementally following the different checkpoints of the course.

---

## Project Topic
**Customer Churn Prediction**

Customer churn prediction aims to identify customers who are likely to stop using a service.  
This problem is commonly addressed using machine learning models trained on customer behavior and subscription data.

---

## Task Definition
This is a **binary classification** problem.

Given a set of customer-related features (e.g. tenure, service usage, contract type, and billing information), the model predicts whether a customer will:
- Churn (leave the service), or
- Not churn (remain a customer)

---

## Dataset Source
The project uses the **Telco Customer Churn dataset**.

- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Data type: tabular dataset
- Target variable: `Churn`

---

## Project Scope & Planned Work

### Checkpoint 1 – Project Setup & Foundations
- GitHub repository setup and collaboration
- Python environment management using **UV**
- Modular project structure
- Data loading and preprocessing
- Runnable baseline training pipeline
- Reproducible setup using `pyproject.toml` and `uv.lock`

### Checkpoint 2 – Code Quality & Experiment Tracking
- Unit testing with `pytest`
- Code quality and formatting tools
- Experiment tracking using **MLflow**
- Logging of model parameters, metrics, and artifacts

### Checkpoint 3 – Model Serving
- Model inference service using **FastAPI**
- REST API endpoint for churn prediction
- Containerization for reproducible execution
- Basic API testing

### Checkpoint 4 – Monitoring & Final Report
- Basic monitoring strategies for the ML system
- Logging and health checks
- Analysis of limitations and future improvements
- Final project report

---

## Team
This project is developed collaboratively by:
- Mourad Mahmoudi  
- Anas Khalil  

---

## How to Run the Project

### Environment Setup
```bash
uv sync
