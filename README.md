# AAPL-stock-pridiction
This project focuses on building a simple yet effective Linear Regression model to predict the stock prices of Apple Inc. (AAPL) using historical stock data. It includes preprocessing, model training, visualization of predictions, and a deployed interactive Streamlit web application.

🔍 Features:
📊 Exploratory Data Analysis (EDA) on Apple stock data
⚙️ Data preprocessing including scaling using MinMaxScaler
🤖 Linear Regression model built with scikit-learn
📉 Visualization of actual vs predicted stock prices
🔮 Future prediction: Forecasts stock prices for the next 1-60 days
🌐 Streamlit web app for interactive visualization and prediction

🚀 Run the App:
- program.py file trains the ML model and create a pkl file for it called as model.pkl. Run `python program.py` to create the pkl file.
- app.py file is for running streamlit local server. This file uses the previously created pkl model file to predict the result. Run `streamlit run app.py` to host the local server.
