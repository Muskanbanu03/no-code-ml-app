# No-Code ML App
A no-code machine learning web app built with Streamlit that automates EDA, preprocessing, model training, hyperparameter tuning, and evaluation — no coding required.

## Features
- Upload your dataset (CSV, Excel, JSON)
- Automatic Exploratory Data Analysis (EDA)
- Data Preprocessing (encoding, scaling, train-val-test split)
- Trains multiple Classification and Regression models
- Hyperparameter tuning for top 4 models
- Final model evaluation on test data with visualizations

## Models Supported
**Classification:** Logistic Regression, Naive Bayes, KNN, Decision Tree, Bagging, Random Forest, AdaBoost, GBM, XGBoost, SVM

**Regression:** Linear Regression, Ridge, Lasso, KNN, Decision Tree, Bagging, Random Forest, AdaBoost, GBM, XGBoost, SVR

## How to Run
```bash
pip install -r requirements.txt
streamlit run ml_app.py
```

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- Matplotlib
- Seaborn

## Author
Muskanbanu03
