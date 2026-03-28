---
title: Predictive Maintenance App
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🚀 Predictive Maintenance System

## 📌 Overview
This project deploys a **machine learning–based predictive maintenance system** that predicts engine condition using real-time sensor inputs.

The application is built using:
- **Streamlit** (UI)
- **Docker** (deployment environment)
- **Hugging Face Spaces** (hosting)
- **Hugging Face Model Hub** (model storage)
- **GitHub Actions** (CI/CD automation)

---

## 🎯 Objective
The goal of this project is to:
- Predict potential engine failures
- Enable proactive maintenance
- Reduce downtime and operational costs

---

## ⚙️ Features

- ✅ User-friendly Streamlit interface  
- ✅ Real-time prediction from sensor inputs  
- ✅ Model loaded dynamically from Hugging Face Model Hub  
- ✅ Automated deployment via GitHub Actions  
- ✅ Dockerized environment for consistent execution  

---

## 🧠 Model Details

- **Model Type:** Ensemble (Random Forest / Boosting)
- **Input Features:**
  - Engine RPM
  - Coolant Temperature
  - Oil Pressure
  - Fuel Pressure
  - Intake Temperature
  - Battery Voltage

- **Target:**
  - Engine Condition (Healthy / Fault Risk)

---

## 🔄 How It Works

1. User inputs sensor values via the UI  
2. Inputs are converted into a pandas DataFrame  
3. Preprocessing pipeline is applied:
   - Validation  
   - Missing value handling  
   - Feature engineering  
4. Model is loaded from Hugging Face Model Hub  
5. Prediction is generated and displayed  

---

## 🏗️ Project Structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── push_to_hf_space.py
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── predict.py
│   ├── preprocess.py
│   └── utils.py
│
└── .github/
    └── workflows/
        └── pipeline.yml
