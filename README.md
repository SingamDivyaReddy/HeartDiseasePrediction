# ❤️ Heart Disease Prediction System

A machine learning-based web application that predicts the risk of heart disease using various health indicators, aimed at providing early diagnosis support for healthcare professionals.

## 🔍 Features

- Predicts heart disease risk using ML models like:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Interactive dashboard for real-time input and results
- Visualizations for data insights using Matplotlib and Seaborn
- User-friendly interface built with **Streamlit**

## 🧠 Tech Stack

- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, Matplotlib, Seaborn
- **Framework:** Streamlit

## 📈 Model Performance

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 85.2%    |
| Random Forest       | 88.7%    |
| SVM (Linear Kernel) | 83.5%    |

> 📊 Accuracy may vary slightly based on dataset splits and preprocessing steps.

## 📦 Installation & Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/SingamDivyaReddy/HeartDiseasePrediction.git
   cd HeartDiseasePrediction
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## 🩺 Use Case

- Helps doctors and patients detect heart disease risk early
- Can be used for **healthcare awareness** and **clinical support tools**
- Supports multiple machine learning algorithms for comparative analysis

## 📊 Dataset

- The model is trained on publicly available heart disease datasets (e.g., UCI repository).
- Features used include: age, blood pressure, cholesterol, chest pain type, etc.

## 👩‍💻 Author

- **Divya Reddy** – [GitHub](https://github.com/SingamDivyaReddy)
