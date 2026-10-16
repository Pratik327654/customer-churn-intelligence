# Customer Churn Prediction & Retention Intelligence System
🚀 **Live Demo:** [Try the Customer Churn Prediction App](https://pratik-churn-ai-2026.streamlit.app)

Predict customer churn risk and explore retention recommendations using a deployed Machine Learning application.

## Project Overview
This project predicts customer churn using Machine Learning and provides rule-based retention recommendations to help businesses identify customers who may be at risk of leaving.

## Key Features
- Customer churn probability prediction
- High-risk and low-risk customer classification
- Logistic Regression model
- Data preprocessing using Scikit-learn pipelines
- SHAP-based model explainability
- Rule-based customer retention recommendations
- Interactive Streamlit web application
- Docker containerization

## Tech Stack
- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- Docker

## Model Performance
On the held-out test set, the Logistic Regression model achieved:

| Metric | Score |
|---|---:|
| Accuracy | 78.21% |
| Precision | 57.70% |
| Recall | 67.11% |
| F1-score | 62.10% |
| ROC-AUC | 84.27% |

These results use a classification threshold of 0.40. Performance may vary with different data and settings.

## Run Locally with Docker

### 1. Clone the repository
```bash
git clone https://github.com/Pratik327654/customer-churn-intelligence.git
cd customer-churn-intelligence
```

### 2. Build the Docker image
```bash
docker build -t customer-churn-app .
```

### 3. Run the application
```bash
docker run --rm -p 8501:8501 customer-churn-app
```

Open `http://localhost:8501` in your browser.

## Project Structure
```text
customer-churn-intelligence/
├── app.py
├── customer_churn_model.pkl
├── Dockerfile
├── requirements.txt
└── README.md
```
## Model Monitoring

The application includes basic prediction monitoring capabilities:

- **Prediction History:** Logs individual churn predictions with timestamps, churn probabilities, risk levels, and thresholds.
- **Prediction Summary:** Displays the number of monitored predictions, high-risk and low-risk counts, and average churn probability.
- **Numeric Feature Deviation Report:** Compares a customer's `tenure`, `MonthlyCharges`, and `TotalCharges` values against reference statistics.
- **Threshold-Based Alerts:** Flags numeric feature deviations when the normalized deviation score exceeds the configured threshold.

**Current limitation:** The numeric deviation report evaluates individual customer inputs against reference statistics. It does not yet provide population-level data drift detection or monitor actual model performance over time. Prediction history is stored in a local CSV file, so persistence on Streamlit Community Cloud is not guaranteed across restarts or redeployments.

## Important Note
Retention recommendations are based on predefined business rules. They are suggestions, not guaranteed interventions or evidence that a customer will churn.

## Project Status

- ✅ Machine Learning model development
- ✅ Docker containerization
- ✅ Automated model testing with pytest
- ✅ CI/CD using GitHub Actions
- ✅ Cloud deployment using Streamlit Community Cloud
- ✅ Prediction monitoring and history tracking
- ✅ Numeric feature deviation report
- ⏳ Population-level data drift monitoring

## Future Improvements

- Implement robust population-level data drift detection
- Track prediction patterns and model performance over time
- Explore automated model retraining when needed

## Author
Pratik Nichal
