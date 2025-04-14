import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Streamlit UI Design
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction using Linear Regression")

# File Upload
uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure Date column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        st.write("### 📊 Data Preview")
        st.write(df.head())

        # Select Feature & Target
        feature = st.selectbox("Select Feature Column", df.columns)
        target = st.selectbox("Select Target Column", df.columns, index=len(df.columns) - 1)

        # Prepare Data
        X = df[[feature]]
        y = df[target]

        # Split stock into train (2012-2018) and test (2019)
        df['Lag_1'] = df['Close'].shift(1)
        df['Lag_7'] = df['Close'].shift(7)

        # Drop missing values
        df.dropna(inplace=True)
        train = df.loc['2012-01-01':'2018-12-31']
        test = df.loc['2019-01-01':]

        X_train = train[['Lag_1', 'Lag_7']]
        y_train = train['Close']
        X_test = test[['Lag_1', 'Lag_7']]
        y_test = test['Close']
        
        # Scale the features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Linear Regression Model
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred_lr = lr_model.predict(X_test_scaled)

        # Create future dates (next 30 days)
        future_dates = pd.date_range(start=X_test.index[-1] + pd.Timedelta(days=1), periods=30, freq="D")

        # Generate future features (assuming `X_test_scaled` has features needed)
        future_features = X_test_scaled[-30:]  # Modify this based on your feature engineering

        # Predict future stock prices
        future_predictions = lr_model.predict(future_features)

        # Convert future predictions to Pandas Series
        future_predictions_series = pd.Series(future_predictions, index=future_dates)


        # Visualization
        st.subheader("📉 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test, label="Actual Prices", color="blue")
        ax.plot(y_test.index, y_pred_lr, label="Predicted Prices", color="red", linestyle="dashed")
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title("Stock Price Prediction (Linear Regression)")
        ax.legend()
        st.pyplot(fig)


        plt.figure(figsize=(12, 6))
        plt.plot(test.index, y_test, label="Actual Prices", color="blue")
        plt.plot(test.index, y_pred_lr, label="Predicted Prices", color="red", linestyle="dashed")
        plt.plot(future_dates, future_predictions_series, label="Future Predictions (30 Days)", color="green", linestyle="dashed")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title("Apple Stock Price Prediction (Linear Regression)")
        plt.legend()
        plt.show()



        # Show 60 days Forecast Table
        future_days = st.slider("📅 select future days to predict", min_value=1, max_value=60, value=30)
        future_dates = pd.date_range(start=X_test.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="D")
        future_features = X_test_scaled[-future_days:] # Modify based on feature engineering
        future_predictions = lr_model.predict(future_features)
        future_predictions_series = pd.Series(future_predictions, index=future_dates)

        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        future_df.set_index("Date", inplace=True)
        st.write(future_df)

        # Plot Future Predictions
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(future_df.index, future_df["Predicted Price"], label="Future Prediction", color="green")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Stock Price")
        ax2.set_title("30-Day Future Stock Price Prediction")
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.error("❌ CSV must contain a 'Date' column for proper visualization.")
