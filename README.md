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

The project uses a Telco Customer Churn dataset provided in YAML format.

- Source: [https://github.com/Anas-kh3/telco-churn-dataset](https://github.com/Anas-kh3/telco-churn-dataset/tree/main)
- Data type: YAML structured dataset
- Target variable: `churn`


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
- Current status is checkpoint 3- Completed.

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
- Miral Jandial
- Pradeep Kumar Reddy Yarragangireddy

---

## How to Run the Project

### Environment Setup
```bash
uv sync
```

## System Architecture
The project follows a modular MLOps architecture that separates data processing, model training, experiment tracking, and model serving.

Main components:

1. **Data Layer**
   - YAML dataset containing customer churn information
   - Data loading and preprocessing scripts

2. **Training Pipeline**
   - Model training implemented in the `src` module
   - Feature processing and model fitting
   - Reproducible training pipeline

3. **Experiment Tracking**
   - Experiments tracked using MLflow
   - Parameters, metrics, and model artifacts logged for each run

4. **Model Storage**
   - Trained models stored in the `models` directory
   - Versioning supported through MLflow artifacts

5. **Model Serving**
   - Inference API built with FastAPI
   - REST endpoint `/predict` for churn predictions

6. **Containerization**
   - Docker used to create a portable and reproducible runtime environment

This architecture ensures modularity, reproducibility, and maintainability of the ML system.
---

## Monitoring & Reliability

Monitoring is an important part of an MLOps system to ensure that deployed models operate correctly.

In this project, basic monitoring capabilities were implemented in the inference service.

### Health Monitoring

A health check endpoint is implemented in the FastAPI service:


```bash
GET /health
```


This endpoint returns the status of the API and allows external systems to verify that the service is running correctly.

Example response:

```json
{
  "status": "ok"
}
```
### Logging

Logging is implemented within the prediction API to monitor model usage and behavior.

The system logs:
- Incoming prediction requests
- Predicted churn probabilities

These logs help developers monitor the system, debug issues, and understand how the model is being used in production.

Together, these monitoring features improve the reliability and observability of the deployed machine learning service.

## Limitations & Future Work

Although the current system demonstrates a complete MLOps pipeline, several improvements could be made in the future.

### Current Limitations

- Monitoring is limited to basic logging and health checks.
- The system does not currently implement automated model retraining.
- Advanced production monitoring such as data drift detection is not implemented.
- Performance benchmarking and load testing of the API are limited.

### Future Improvements

Future work could include:

- Implementing automated CI/CD pipelines for model deployment
- Adding advanced monitoring tools such as Prometheus or Grafana
- Implementing data drift detection and model performance monitoring
- Adding automated retraining pipelines when model performance decreases
- Scaling the API using container orchestration platforms such as Kubernetes

These improvements would make the system more robust and production-ready.

