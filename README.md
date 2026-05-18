# Multilinear Regression using SGD

A Streamlit web application that performs Multilinear Regression using Stochastic Gradient Descent (SGD). This app allows users to upload their own datasets, train a regression model interactively, and make real-time predictions.

## Features

- **Upload CSV Data**: Easily upload any numerical dataset in CSV format.
- **Dynamic Configuration**: Use the sidebar to select your target variable and tune model hyperparameters like `Max Iterations` and `Learning Rate`.
- **Model Evaluation**: Automatically calculates and displays the Mean Squared Error (MSE) and R² Score.
- **Visualizations**: View a scatter plot comparing the actual vs. predicted values to assess model performance.
- **Real-time Predictions**: Dynamically generated input forms allow you to enter custom feature values and get instant predictions.

## Tech Stack

- **Python**
- **Streamlit** (for the interactive web interface)
- **Scikit-Learn** (for the `SGDRegressor`, `StandardScaler`, and metrics)
- **Pandas & NumPy** (for data manipulation)

## Installation & Usage

1. **Clone the repository** (if applicable) or download the project files.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
5. **Open in Browser**: The app will automatically open in your default browser at `http://localhost:8501`.

## Example Data

A sample file `student_data.csv` is included to test the application. It contains features like hours studied, sleep hours, previous scores, and attendance to predict a student's performance index.
