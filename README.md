🏆 HackVerse 2026 – Official Submission

<div align="center">
🥷 Team Bit-Ninjas
Med-Bot Mini 2.0
AI-Powered Multi-Disease Screening Robot
</div>

## 📌 Overview

Med-Bot Mini 2.0 is an AI-driven, portable healthcare screening robot designed to enable early risk detection in underserved and resource-limited environments.

Access to preventive healthcare remains a major challenge in rural and semi-urban regions. Most individuals seek medical assistance only after symptoms become severe, resulting in delayed diagnosis and increased treatment costs.

Med-Bot Mini 2.0 addresses this gap by integrating:

Biomedical sensor monitoring

Computer vision-based health analysis

Speech and neurological screening

AI-powered multimodal risk scoring

The system performs real-time preliminary health screening using Vision + Vitals + Voice AI Fusion, providing an intelligent risk classification (Low / Moderate / High).

## 🎯 Problem Statement

Preventive health screening is often expensive and inaccessible.

Wearable devices monitor only limited parameters.

Hospitals are overcrowded for minor conditions.

There is no affordable multi-parameter AI screening platform in a portable robotic form.

## 💡 Proposed Solution

Med-Bot Mini 2.0 combines:

Vital sign monitoring

Computer vision health analysis

Speech-based neurological screening

Machine learning risk assessment

into a single, intelligent robotic healthcare assistant.

Heavy AI computation runs on a connected laptop server, while the Raspberry Pi-based robot acts as an intelligent data acquisition and interaction interface.

## ⚙️ Core Features

# 🫀 Vital Monitoring

Heart Rate Monitoring (MAX30102)

SpO₂ Detection

Non-Contact Temperature (MLX90614)

Skin Hydration Estimation

# 🧠 Neurological Screening (FAST Protocol)

Face symmetry detection

Arm drift detection

Speech clarity analysis

# 😮‍💨 Respiratory Monitoring

Breathing rate detection (camera + audio)

Oxygen level fusion risk scoring

# 🦴 Musculoskeletal Analysis

Posture detection

Exercise form correction

Joint imbalance detection

# 😌 Mental Health Monitoring

Emotion detection

Stress & fatigue analysis (NeuroBand AI)

# 💊 Medicine Identification

OCR-based medicine recognition

Side-effect warning system

# 🧪 Technical Architecture

Hardware Stack

Raspberry Pi Zero 2 W

MAX30102 (Heart Rate & SpO₂ Sensor)

MLX90614 (Infrared Temperature Sensor)

Skin Moisture Sensor

Raspberry Pi Camera Module

Microphone & Speaker

OLED Display

Servo Motors

Software Stack

Python

OpenCV

MediaPipe

Scikit-learn / XGBoost

LLM-based symptom reasoning

SQLite Database

Socket Communication (WiFi-based Pi ↔ Laptop Server)

# 🔄 System Workflow

User initiates screening via voice command.

Robot collects biomedical sensor data.

Camera captures facial and posture data.

Data is transmitted to the AI server.

AI modules analyze multimodal inputs.

Risk scoring engine fuses outputs.

Robot announces and displays results.

# 📊 Innovation Highlights

Multimodal AI Fusion (Vision + Vitals + Voice)

Early stroke screening using FAST protocol

Preventive healthcare orientation

Modular and scalable architecture

Operates without requiring a smartphone application

# 🛠 Setup & Installation Guide

1. Clone Repository
```bash
git clone https://github.com/your-team/medbot-mini.git
cd medbot-mini
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
For Raspberry Pi:
```bash
sudo apt update
sudo apt install python3-opencv
```
3. Start AI Server (Laptop)
```bash
python server.py
```
4. Start Robot Client (Raspberry Pi)
```bash
python client.py
```
⚠️ Disclaimer

Med-Bot Mini 2.0 is a preliminary health screening system and is not intended to replace professional medical diagnosis. It provides early risk indications for awareness and preventive action only.

# 🚀 Future Scope

Edge AI optimization (fully running on Raspberry Pi)

Cloud-based health data storage

Remote doctor consultation integration

Addition of blood pressure monitoring module

IoT-enabled health monitoring network
