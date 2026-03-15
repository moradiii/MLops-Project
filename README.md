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
- Target variable: churn

---

## Project Scope & Planned Work

### Checkpoint 1 – Project Setup & Foundations

- GitHub repository setup and collaboration
- Python environment management using **UV**
- Modular project structure
- Data loading and preprocessing
- Runnable baseline training pipeline
- Reproducible setup using pyproject.toml and uv.lock

### Checkpoint 2 – Code Quality & Experiment Tracking

- Unit testing with pytest
- Code quality and formatting tools
- Experiment tracking using **MLflow**
- Logging of model parameters, metrics, and artifacts

### Checkpoint 3 – Model Serving

- Model inference service using **FastAPI**
- REST API endpoint for churn prediction
- Containerization for reproducible execution
- Basic API testing - Current status is checkpoint 3- Completed.

### Checkpoint 4 – Monitoring & Final Report

- Basic monitoring strategies for the ML system
- Logging and health checks
- Analysis of limitations and future improvements
- Final project report

## 5. Monitoring & Reliability

To improve the reliability and observability of the system, we implemented a basic monitoring strategy adapted to the scope of the project.

### Health Check

A /health endpoint was added to verify that the API is running correctly and that the model is successfully loaded.

## System Architecture

The system follows an end-to-end MLOps pipeline:

Dataset
↓
Preprocessing
↓
Model Training
↓
MLflow Experiment Tracking
↓
Model Artifact
↓
FastAPI Inference Service
↓ Docker Container

## MLOps Practices

This project follows several MLOps best practices:

- **Version Control**: Git and GitHub were used to track code changes and collaborate between team members.
- **Environment Management**: UV manages project dependencies through pyproject.toml and uv.lock.
- **Testing**: Unit tests ensure reliability of data loading, preprocessing, and training logic.
- **Experiment Tracking**: MLflow logs model parameters, metrics, and artifacts.
- **Model Serving**: A FastAPI service exposes the trained model through a REST API.
- **Containerization**: Docker packages the application for reproducible deployment.
- **Monitoring**: Health checks, logging, and metrics endpoints provide basic monitoring.

## Limitations & Future Work

Although the system demonstrates a complete MLOps pipeline, several improvements could be made.

Limitations:

- Limited dataset size
- Basic monitoring implementation
- Local deployment only

Future work:

- Deploy the service to a cloud platform
- Add Prometheus and Grafana for advanced monitoring
- Implement automated retraining pipelines
- Add data drift and model drift detection

---

## Team

This project is developed collaboratively by:

- Mourad Mahmoudi
- Anas Khalil
- Miral Jandial
- Pradeep Kumar Reddy Yarragangireddy

---

```bash
uv sync
```
