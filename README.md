# AI in Market Trend Analysis

An end-to-end **AI-powered market intelligence platform** that analyzes historical sales data and customer reviews to generate actionable business insights.  
The system integrates **time-series forecasting, deep learning, NLP-based sentiment analysis, and anomaly detection**, deployed through an interactive **Streamlit dashboard**.

🔗 **Live Demo:** https://ai-in-market-trend-analysis.streamlit.app/

---

## 📌 Project Overview

Understanding market behavior is critical for effective business decision-making. This project applies Artificial Intelligence techniques to:

- Forecast future sales trends
- Analyze the impact of promotions
- Understand customer sentiment from reviews
- Detect anomalies in sales and sentiment patterns
- Provide interpretable, business-ready insights

The project is developed as part of **Module E: AI Applications – Individual Open Project**.

---

## 🚀 Key Features

- **Demand Forecasting**
  - Prophet-based statistical forecasting
  - LSTM deep learning model for non-linear temporal patterns
- **Promotion Impact Analysis**
  - External regressor integration to model promotion effects
- **Customer Sentiment Analysis**
  - TF-IDF + Logistic Regression for review sentiment classification
- **Anomaly Detection**
  - Isolation Forest for identifying unusual sales and sentiment behavior
- **Interactive Dashboard**
  - User-friendly Streamlit app for exploration and insights

---

## 🧠 AI & ML Techniques Used

- **Time Series Forecasting:** Prophet, LSTM (Keras)
- **Natural Language Processing:** TF-IDF, Logistic Regression
- **Anomaly Detection:** Isolation Forest
- **Evaluation Metrics:** MAPE, MAE, RMSE, Accuracy, Confusion Matrix

---

## 🗂️ Repository Structure
AI-in-Market-Trend-Analysis/
│
├── AI in Market Trend Analysis Final.ipynb # Main evaluation notebook
├── app1.py # Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
│
├── prophet_demand_forecast.pkl # Trained Prophet model
├── lstm_demand_forecast.keras # Trained LSTM model
├── feature_scaler.pkl # Scaler for LSTM features
├── sentiment_model.pkl # Sentiment classification model
└── tfidf_vectorizer.pkl # TF-IDF vectorizer

---

## 📊 Datasets Used

- **Store Sales Time Series Dataset (Kaggle)**
  - Used for demand forecasting and trend analysis
- **Amazon Fine Food Reviews Dataset (Kaggle)**
  - Used for customer sentiment analysis

All datasets are publicly available and anonymized.

---

## ⚙️ How to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/suryacharan945/AI-in-Market-Trend-Analysis.git
cd AI-in-Market-Trend-Analysis
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Run the Notebook
```
AI in Market Trend Analysis Final.ipynb
```
4️⃣ Run Streamlit App
```
streamlit run app1.py
```
📈 Results & Insights

Prophet and LSTM models effectively capture seasonal and long-term sales trends

Promotion intensity improves forecasting stability

Sentiment model achieves high precision for positive and negative classes

Anomaly detection highlights demand spikes, drops, and sentiment shifts

The system provides strong decision support for inventory, pricing, and marketing strategies

⚖️ Ethical Considerations

Uses publicly available, anonymized datasets

Predictions are intended as decision-support, not automated decision-making

Model limitations and data biases are acknowledged

Designed with transparency and responsible AI principles

🔮 Future Enhancements

Integration of external factors (economic indicators, holidays, weather)

Transformer-based forecasting models

BERT-based sentiment analysis

Real-time data ingestion via APIs

Explainable AI techniques for improved transparency
