import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="SGD Regression", layout="wide")
st.title("Multilinear Regression using SGD")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Handle missing values
    data = data.dropna()
    
    # Select only numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("The uploaded CSV does not contain any numeric columns.")
    else:
        with st.expander("View Dataset", expanded=False):
            st.dataframe(data, use_container_width=True)

        st.sidebar.subheader("Model Settings")
        target_col = st.sidebar.selectbox("Select Target Variable", numeric_cols, index=len(numeric_cols)-1)
        
        max_iter = st.sidebar.slider("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
        learning_rate = st.sidebar.number_input("Learning Rate (eta0)", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        
        features = [col for col in numeric_cols if col != target_col]
        
        if not features:
            st.error("Not enough numeric features available for training.")
        else:
            X = data[features]
            y = data[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SGDRegressor(
                max_iter=max_iter,
                eta0=learning_rate,
                learning_rate='constant',
                random_state=42
            )

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            col2.metric("R² Score", f"{r2:.4f}")

            st.subheader("Actual vs Predicted")
            chart_data = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
            st.scatter_chart(chart_data)

            st.subheader("Make a Prediction")
            with st.form("prediction_form"):
                st.write("Enter feature values:")
                input_data = []
                cols = st.columns(min(len(features), 4))
                for i, col in enumerate(features):
                    with cols[i % len(cols)]:
                        value = st.number_input(f"{col}", value=float(X[col].mean()), format="%.2f")
                        input_data.append(value)
                
                submitted = st.form_submit_button("Predict")
                if submitted:
                    input_array = np.array([input_data])
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)
                    st.success(f"Predicted {target_col}: {prediction[0]:.4f}")

else:
    st.info("Please upload a CSV file from the sidebar to get started.")