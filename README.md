# 🔥 Enhancing Building Safety through Machine Learning and Deep Learning Based Smoke Detection [C169]

> **Major Project — Bachelor of Engineering in Computer Science and Engineering (AIML)**  
> Lords Institute of Engineering and Technology (UGC Autonomous), Hyderabad  
> Academic Year 2025–2026

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.2-green?logo=django)](https://djangoproject.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange?logo=tensorflow)](https://tensorflow.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)

---

## 📌 Project Overview

This project presents an intelligent building safety system that detects fire and smoke using a dual-module approach:

- **ML Module** — Seven classical machine learning classifiers trained on a real-world IoT sensor dataset (62,630 readings, 13 sensor channels) achieve near-perfect AUC-ROC scores above 0.999.
- **DL Module** — A MobileNetV2 CNN trained via transfer learning achieves 96.98% validation accuracy on fire/no-fire image classification. A YOLOv8 object detection model draws bounding boxes around fire and smoke regions on the same prediction page.

The entire system is delivered as a full-stack Django web application accessible through any browser.

---

## 👨‍💻 Team

| Name | Roll Number |
|------|-------------|
| Syed Abdul Wasay | 160922748015 |
| M.A.Omer | 160922748048 |
| Syed Afeef ul Luqman | 160922748037 |
| Mohammed Muneebuddin Ahmed | 160922748060 |

**Project Guide:** Dr. Mohammed Tajuddin, Associate Professor  
**Co-Guide / HoD:** Dr. Abdul Rasool MD, Associate Professor & Head of Department, CSE (AIML)  
**Institution:** Lords Institute of Engineering and Technology, Hyderabad

---

## 🏗️ Project Structure

```
Buliding_Saftey_Through_Machine_learning/
│
├── admins/                     # Admin app (login, dashboard, user management)
├── users/                      # Users app (dataset, training, prediction, CNN+YOLO)
├── templates/                  # HTML templates
│   ├── admins/
│   └── users/
├── media/                      # Model files and dataset
│   ├── models/                 # Trained ML classifiers (.pkl) + YOLO (best.pt)
│   ├── cnn_model.h5            # Trained MobileNetV2 CNN
│   ├── cnn_classes.json        # Class index mapping
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── smoke_detection_iot.csv # IoT sensor dataset (62,630 rows)
│   └── cnn_dataset/            # fire/ and no_fire/ image folders
├── static/                     # Static CSS/JS/images
├── manage.py
├── requirements.txt
├── Procfile                    # Railway/Gunicorn deployment
├── train_cnn.py                # Standalone CNN training script
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 User Registration & Login | Role-based access with admin activation workflow |
| 📊 Dataset Browser | Browse all 62,630 IoT sensor rows with pagination and search |
| 🤖 ML Training | Train all 7 classifiers on demand with live metrics and charts |
| 📡 Sensor Prediction | Enter 13 sensor values → instant smoke detection result |
| 🖼️ CNN Image Classification | Upload image → MobileNetV2 fire/no-fire result with confidence |
| 🔲 YOLO Fire & Smoke Detection | YOLOv8 draws bounding boxes around fire & smoke on same image |
| 👤 Admin Dashboard | Activate/deactivate user accounts |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Django 5.2 |
| ML Library | Scikit-Learn 1.6.1 |
| Deep Learning | TensorFlow 2.10 / Keras |
| Object Detection | Ultralytics YOLOv8 |
| Computer Vision | OpenCV |
| Data Processing | Pandas, NumPy 1.26.4 |
| Frontend | Bootstrap 5, Chart.js |
| Database | SQLite (Django ORM) |
| Deployment | Railway + Gunicorn + WhiteNoise |

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Building-Safety-Smoke-Detection.git
cd Building-Safety-Smoke-Detection/Buliding_Saftey_Through_Machine_learning
```

### 2. Create and activate conda environment
```bash
conda create -n smoke python=3.10
conda activate smoke
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Apply migrations
```bash
python manage.py migrate
```

### 5. Train the CNN model (first time only)
```bash
python train_cnn.py
```

### 6. Run the development server
```bash
python manage.py runserver
```

Visit: **http://127.0.0.1:8000/**

---

## 🔑 Default Admin Credentials

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin` |

---

## 📈 ML Model Performance

| Model | Precision | Recall | AUC-ROC |
|-------|-----------|--------|---------|
| Random Forest | ~100% | ~100% | ~100% |
| Gradient Boosting | ~100% | ~100% | ~100% |
| AdaBoost | ~99.9% | ~99.9% | ~100% |
| Logistic Regression | ~99.3% | ~99.0% | ~99.9% |
| SVM | ~100% | ~99.9% | ~100% |
| Decision Tree | ~99.9% | ~99.9% | ~99.9% |
| KNN | ~100% | ~100% | ~100% |

**CNN Validation Accuracy:** 96.98% (MobileNetV2, Kaggle Fire Dataset)

---

## 🌐 Live Demo

🔗 **[Deployed on Railway](https://building-safety-smoke-detection-production.up.railway.app/)** ← update this link after deployment

---

## 📄 License

This project is submitted as a Major Project for the partial fulfillment of the award of Bachelor of Engineering at Lords Institute of Engineering and Technology, Hyderabad. All rights reserved by the project team.
