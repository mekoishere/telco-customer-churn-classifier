# Telco Customer Churn Predictor

This repository contains a complete Machine Learning project aimed at predicting customer churn in a telecommunications company. It includes a model training script with rigorous hyperparameter tuning and a user-friendly interactive web application built with Streamlit, which can be easily deployed using Docker.

## The Challenge: Imbalanced Data & Optimizing for Recall

In the telecommunications industry, the number of customers who stay with the company is vastly larger than the number of customers who leave. 

**The Problem:** If a model simply predicts that everyone will stay, it might achieve a very high overall accuracy, but it completely fails at its business purpose: identifying the customers who are actually leaving.

**The Solution:** To build a truly useful business tool, this project explicitly focuses on identifying the churning customers. 
* **Class Weighting:** The model is initialized with `class_weight='balanced'` to heavily penalize the model for missing churning customers.
* **Maximizing Recall:** During hyperparameter tuning via `GridSearchCV`, the model is strictly optimized for Recall on Class 1. This ensures that the final model prioritizes catching as many departing customers as possible, even at the cost of a few false positives. It is much more cost-effective to offer a retention discount to a customer who wasn't going to leave than to completely lose a customer because the model missed them.

## Features
* **Automated Data Preprocessing:** Handles missing numerical values and maps categorical variables using One-Hot Encoding.
* **Advanced ML Modeling:** Uses `HistGradientBoostingClassifier` with cross-validated grid search.
* **Interactive UI:** A Streamlit dashboard allowing users to input customer demographics and service details via sliders and dropdowns.
* **Real-time Prediction:** Calculates the absolute churn risk and displays the probability percentage.
* **Prediction Logging:** Automatically saves the history of predictions to a local csv file for future monitoring and analysis.
* **Containerized Deployment:** Fully supported by Docker for isolated and consistent execution.

## Application Preview

The Streamlit web application interface.

<img width="1083" height="654" alt="Zrzut ekranu 2026-06-16 121630" src="https://github.com/user-attachments/assets/34e7f5a2-4d73-4be3-87a9-aee74e5ab2ca" />

## Tech Stack
* Python
* Pandas - Data manipulation
* Scikit-Learn - Machine Learning & preprocessing
* Streamlit - Web interface
* Joblib - Model saving/loading
* Docker - Containerization

## How to Run the Project

### Prerequisites
Make sure you have Docker installed and running on your machine.

### Running with Docker

1. Build the Docker image by running:
```bash
docker build -t churn-app .
```

2. Once the build is complete, run the container:
```bash
docker run -p 8501:8501 churn-app
```

3. Go to `http://localhost:8501` to access the application.

## File Structure
* `Dockerfile` - Instructions for building the Docker container.
* `main.py` - Script for data preprocessing, training, and optimizing the ML model.
* `app.py` - The Streamlit frontend application.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv` - The Telco dataset.
* `churn_model.pkl` - The trained ML model generated after running the train script.
* `model_columns.pkl` - Saved column structure to ensure consistent input shapes during prediction.
* `churn_logs.csv` - File storing prediction history.
