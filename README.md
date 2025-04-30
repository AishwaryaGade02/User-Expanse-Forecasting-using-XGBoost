# 📈 User Expense Forecasting with PySpark and XGBoost

This project demonstrates an end-to-end pipeline for **forecasting user expenses** using time series modeling. It integrates PySpark for scalable preprocessing and feature engineering, and leverages **XGBoost** for modeling transaction-level financial data.

---

## 🧠 Problem Statement

Predict future daily user expense patterns using transaction data, card and user metadata, and derived temporal and lag-based features.

---

## 🧰 Tech Stack

- **Language**: Python
- **Distributed Processing**: Apache Spark (PySpark)
- **Data Source**: [Kaggle Fraud Dataset](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets) via `kagglehub`
- **Modeling**: XGBoost Regressor
- **Visualization**: matplotlib
- **Libraries**: pandas, scikit-learn, numpy, seaborn

---

## 🔄 Workflow

1. **Data Acquisition**  
   - Download Kaggle dataset using `kagglehub`
   - Load and merge transactions, user, and card data

2. **Data Preprocessing (PySpark)**  
   - Handle missing values  
   - Remove fraudulent transactions  
   - Clean currency fields (`$`)  
   - Derive date-based and cyclic features (`sin/cos`)  
   - Compute lag-based predictors (`lag_1`, `lag_7`, etc.)

3. **Feature Engineering**  
   - Group by date to get daily total spending  
   - Add `year`, `day`, and `weekday` indicators  
   - Normalize features using `StandardScaler`

4. **Modeling**  
   - Use `TimeSeriesSplit` with a gap for realistic validation  
   - Train XGBoost regressor across hyperparameter grid  
   - Select best model using average RMSE

5. **Evaluation**  
   - Plot predicted vs actual values for both train and test sets  
   - Report RMSE for model performance

---

## 📊 Visual Outputs

- **Time Series Plot** of daily spending
- **Actual vs Predicted Plots** for train and test sets
- **Lag and Seasonality Features** visualized

---

## ✅ Model Highlights

- Robust time-aware validation using `TimeSeriesSplit`
- Realistic financial behavior modeled using lagged features and seasonality
- Final model trained with optimal `max_depth`, `learning_rate`, and `subsample`

