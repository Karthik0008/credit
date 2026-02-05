# 💳 Credit Scoring System

An intelligent credit risk assessment application built with Streamlit and Machine Learning.

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
