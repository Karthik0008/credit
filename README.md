# 💳 Credit Scoring System

An intelligent credit risk assessment application built with Streamlit and Machine Learning.

This repository contains an AI‑powered Credit Scoring and Default Risk Prediction System built with Python, Scikit‑learn and Streamlit. The app calculates credit scores on a 300–850 scale, predicts default probability, and segments customers into risk tiers (Excellent, Very Good, Good, Fair, Poor). It includes an interactive analytics dashboard, batch scoring for thousands of customers, an interest‑rate calculator, and personalized recommendations to help improve credit health.

**Live Demo:** https://credit--scoring.streamlit.app/

## ✨ Key Features

1. Supervised ML model for credit‑default prediction and credit scoring
2. 5‑factor score (payment history, utilization, age, mix, inquiries) mapped to 300–850 range
3. Interactive Streamlit UI with banking‑style dashboard and KPI cards
4. Analytics Dashboard for portfolio‑level insights (score distribution, tiers, risk levels)
5. Recommendations page with customer‑specific improvement suggestions
6. Batch Scoring: upload CSV to score thousands of customers at once
7. Interest Rate Calculator that prices loans dynamically based on risk tier and DTI

## 🛠 Tech Stack
- **Language:** Python
- **ML & Data:** Scikit‑learn, Pandas, NumPy
- **App Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib/Seaborn
- **Deployment:** Streamlit Cloud / Flask API

## 🚀 How to Run

### Prerequisite
Ensure you have **Python 3.8+** installed on your system.

### 1. Setup Environment
Open your terminal or command prompt in this directory (`c:\Users\DELL\Desktop\credit`).

It is recommended to create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages using the provided requirements file:
```bash
pip install -r requirement.txt
```
*Note: The file is named `requirement.txt` (singular), not `requirements.txt`.*

### 3. Run the Application
Start the Streamlit web application:
```bash
python -m streamlit run streamlit_app.py
```
*(Using `python -m streamlit` is recommended to avoid PATH issues)*

The application will automatically open in your default web browser at `http://localhost:8501`.

## 📂 Project Structure
- `streamlit_app.py`: Main application code.
- `data/`: Contains dataset files (`train.csv`, `preprocessed_data.csv`).
- `models/`: Contains the trained ML model (`credit_scoring_model.pkl`) and scaler.
- `requirement.txt`: List of Python dependencies.

## 🛠 Troubleshooting
- **Model Not Found:** Ensure `credit_scoring_model.pkl` is inside the `models/` directory.
- **Port In Use:** If port 8501 is busy, Streamlit will try the next available port (8502, etc.). Check the terminal output for the correct URL.
