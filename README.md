# MLOps-Project

## Project description
This project implements a small end-to-end ML pipeline (data loading → preprocessing → training → evaluation) following MLOps best practices (reproducible environment, clean structure, Git tracking).

## Task definition
Multi-class classification: predict the class of a flower using tabular features.

## Dataset source
Iris dataset from `scikit-learn` (`sklearn.datasets.load_iris`).

## Team roles
- Mourad Mahmoudi: project setup, UV environment, repository structure, baseline pipeline
- Teammate: documentation/README, testing & quality checks (pre-commit + unit tests) and later API/Docker

## How to run
### Setup
```bash
uv sync
