# Online News Popularity Prediction

## Overview:

This project focuses on predicting online news popularity using regression machine learning models. The repository contains files and scripts for data preprocessing, exploratory data analysis, feature engineering, and model training.

## Files:

- **.github/workflows:** Directory containing GitHub Actions workflows for continuous integration.

- **templates:** Directory containing HTML templates for the web application.

- **Dockerfile:** Configuration file for Docker, facilitating containerization of the application.

- **LICENSE:** Project license details.

- **Procfile:** Configuration file for Heroku deployment.

- **README.md:** The main documentation for the project.

- **Shares.ipynb:** Jupyter Notebook providing an in-depth exploration and analysis of online news popularity prediction.

- **app.py:** Python script for running the web application using Flask.

- **best_xgb_model.pkl:** Pickle file containing the best-trained XGBoost model for predicting online news popularity.

- **requirements.txt:** Python package dependencies for traditional virtual environments.

- **standard_scaler.pkl:** Pickle file containing a trained standard scaler for feature scaling.

## Usage:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Popular-News.git
   cd Popular-News
   ```

2. **Setup Environment:**
   - Using traditional virtual environments e.g conda,pipenv:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

4. **Explore Notebooks:**
   - Dive into the `Shares.ipynb` notebook for a detailed analysis of online news popularity prediction.

## Deployment:

The online news popularity prediction application can be containerized using Docker. Build the Docker image using the provided Dockerfile and deploy it to your preferred platform. The Procfile is included for Heroku deployment.

## Dataset:

- **Source:** The dataset used for online news popularity prediction can be downloaded from Kaggle.

## Models:

- **XGBoost Model (best_xgb_model.pkl):**
  - The best-trained XGBoost model for online news popularity prediction.

## Preprocessing:

- **Standard Scaler (standard_scaler.pkl):**
  - A trained standard scaler for feature scaling.

## Libraries and Tools:

- **Pandas:** Data manipulation and analysis.

- **Matplotlib and Seaborn:** Data visualization.

- **NumPy:** Numerical operations.

- **Scikit-learn:** Machine learning library for preprocessing, modeling, and evaluation.

- **XGBoost:** Gradient boosting library for regression.

- **Optuna:** Hyperparameter optimization library.

## Flask Web Application:

- The web application is built using Flask.

## License:

This project is licensed under the [MIT License](LICENSE).
