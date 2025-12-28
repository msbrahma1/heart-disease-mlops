# Heart Disease Risk Prediction - MLOps Pipeline
## Final Comprehensive Report

---

## Table of Contents
1. Executive Summary
2. Project Overview & Objectives
3. Dataset & Exploratory Data Analysis
4. Feature Engineering & Preprocessing
5. Model Development & Training
6. Experiment Tracking with MLflow
7. MLOps Architecture & Deployment
8. Results & Model Performance
9. Implementation Summary
10. Conclusion & Future Work

---

## 1. Executive Summary

This report documents a complete end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease risk from patient health data. The project successfully demonstrates modern MLOps practices including data acquisition, exploratory analysis, feature engineering, model development, experiment tracking, containerization, and production deployment.

**Key Achievements:**
- Built and evaluated two classification models (Logistic Regression: 85.25% accuracy, Random Forest: 86.89% accuracy)
- Implemented MLflow for comprehensive experiment tracking with 3 complete runs logged
- Created production-ready Docker containers with health checks and auto-scaling
- Designed Kubernetes deployment manifests for cloud-native deployment
- Established CI/CD pipeline with GitHub Actions (all tests passing)
- Implemented comprehensive monitoring and logging infrastructure
- Achieved 100% test coverage with 10 unit test cases

**Business Impact:**
The final Random Forest model achieves 86.89% accuracy with 94.59% ROC-AUC, making it suitable for deployment in clinical decision support systems. The MLOps infrastructure ensures reproducibility, scalability, and maintainability for continuous improvement.

---

## 2. Project Overview & Objectives

### 2.1 Project Goals

1. **Data Acquisition:** Obtain and prepare the UCI Heart Disease dataset (303 samples, 13 clinical features)
2. **Exploratory Analysis:** Understand feature distributions, correlations, and class balance
3. **Feature Engineering:** Prepare features through scaling and imputation
4. **Model Development:** Train and evaluate multiple classification algorithms
5. **Experiment Tracking:** Log all experiments with parameters, metrics, and artifacts
6. **Reproducibility:** Ensure full reproducibility through containerization and packaging
7. **CI/CD Pipeline:** Automate testing and validation with GitHub Actions
8. **Containerization:** Build Docker images for production deployment
9. **Production Deployment:** Design Kubernetes manifests for cloud deployment
10. **Monitoring:** Implement logging and monitoring infrastructure

### 2.2 Methodology

- **Data Source:** UCI Machine Learning Repository - Heart Disease Dataset (Cleveland)
- **ML Framework:** Scikit-learn for model training and evaluation
- **Experiment Tracking:** MLflow for parameter/metric logging
- **Containerization:** Docker for environment isolation
- **Orchestration:** Kubernetes manifest (k8s-deployment.yaml)
- **CI/CD:** GitHub Actions with pytest and flake8
- **Environment:** Google Colab for development, GitHub for version control

### 2.3 Team & Tools

- **Primary Language:** Python 3.9
- **Key Libraries:** pandas, numpy, scikit-learn, MLflow, Flask, Gunicorn
- **Version Control:** Git & GitHub
- **Continuous Integration:** GitHub Actions
- **Containerization:** Docker & Docker Compose
- **Orchestration:** Kubernetes (Minikube/GKE compatible)

---

## 3. Dataset & Exploratory Data Analysis

### 3.1 Dataset Overview

| Attribute | Value |
|-----------|-------|
| Dataset | UCI Heart Disease (Cleveland) |
| Total Samples | 303 |
| Features | 13 clinical attributes |
| Target | Binary (0: No disease, 1: Disease present) |
| Missing Values | ~5% (handled via imputation) |
| Class Balance | 164 No Disease (54%), 139 Disease (46%) |

### 3.2 Feature Description

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| age | Numeric | 29-77 | Age in years |
| sex | Binary | 0-1 | 0=Female, 1=Male |
| cp | Categorical | 1-4 | Chest pain type |
| trestbps | Numeric | 94-200 | Resting blood pressure (mm Hg) |
| chol | Numeric | 126-564 | Serum cholesterol (mg/dl) |
| fbs | Binary | 0-1 | Fasting blood sugar > 120 mg/dl |
| restecg | Categorical | 0-2 | Resting ECG results |
| thalach | Numeric | 60-202 | Maximum heart rate achieved |
| exang | Binary | 0-1 | Exercise induced angina |
| oldpeak | Numeric | 0-6.2 | ST depression induced by exercise |
| slope | Categorical | 1-3 | Slope of peak exercise ST segment |
| ca | Numeric | 0-3 | Major vessels colored by fluoroscopy |
| thal | Categorical | 3-7 | Thalassemia type |

### 3.3 EDA Findings

**Class Distribution:**
- No Disease: 164 samples (54.1%)
- Disease Present: 139 samples (45.9%)
- Balanced dataset suitable for classification

**Feature Correlations:**
- Strongest positive correlations with disease: ca (0.44), cp (0.33), exang (0.29)
- Strongest negative correlations: thalach (-0.42), trestbps (-0.14)
- Age shows weak correlation (0.22) with disease

**Missing Values:**
- ca (4 missing), thal (2 missing)
- Handled via mean imputation during preprocessing

**Data Quality:**
- No duplicates detected
- No obvious outliers beyond expected ranges
- Data well-suited for ML analysis

---

## 4. Feature Engineering & Preprocessing

### 4.1 Data Preprocessing Pipeline

