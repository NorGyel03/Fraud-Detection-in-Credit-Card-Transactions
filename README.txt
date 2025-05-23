############################################################ 💳 Credit Card Fraud Detection ########################################################################

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It uses a real-world dataset and addresses the challenge of class imbalance through resampling and robust modeling.

## 📂 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- The dataset contains transactions made by European cardholders in September 2013.
- It includes 284,807 transactions, out of which only 492 are fraudulent (highly imbalanced).

## 🛠 Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Streamlit (for dashboard)
- Joblib (for model saving/loading)

## 🚀 Steps Involved

1. **Data Preprocessing**

2. **Model Training**

3. **Model Saving**

4. **Streamlit Dashboard**

5. **Deployment**
   - Run app locally using:  
     ```bash
     streamlit run app.py

## 📊 Output Visualizations
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Fraud Prediction Results (via Streamlit UI)

## ✅ Conclusion
This project demonstrates how to build an end-to-end fraud detection system with machine learning. It tackles class imbalance using SMOTE, uses XGBoost for modeling, and provides an interactive dashboard for testing and visualization.

## 📎 Notes
- Ensure `scaler.pkl` and `model.pkl` are present in the app directory when running `app.py`
- Use `joblib.load()` to load them in your Streamlit app

