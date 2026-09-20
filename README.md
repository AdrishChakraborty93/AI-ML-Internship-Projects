# AI & ML Internship Projects

This repository contains projects completed as part of my Artificial Intelligence
and Machine Learning internship.

## Task 1 – California Housing Price Prediction

### Objective

Build a Linear Regression model to predict median house values using the
California Housing dataset and demonstrate the complete Machine Learning workflow.

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab

### Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Preprocessing
4. Train-Test Split
5. Linear Regression Model Training
6. Prediction
7. Model Evaluation
8. Data Visualization
9. Model Saving

### Dataset

California Housing dataset provided by Scikit-learn.

### Model

Linear Regression

### Results

| Metric | Result |
|---|---:|
| MAE | 0.533 |
| RMSE | 0.746 |
| R² Score | 0.576 |

### Files

- `California_Housing_Linear_Regression.ipynb` – Complete Google Colab notebook
- `California_Housing_Linear_Regression_Report.pdf` – Project report

### Future Improvements

Possible improvements include:

- Feature engineering
- Ridge and Lasso Regression
- Random Forest Regression
- Gradient Boosting
- Cross-validation
- Hyperparameter tuning

  ## Task 2 – Feature Scaling, Model Optimization & Performance Comparison

### Objective

Compare multiple regression models for California housing price prediction
and evaluate their performance using RMSE and R².

### Models Used

- Linear Regression
- Ridge Regression
- Decision Tree Regressor

### Machine Learning Workflow

1. Load the California Housing dataset
2. Separate features and target
3. Perform train-test split
4. Apply feature scaling using StandardScaler
5. Train multiple regression models
6. Compare model performance
7. Select the best-performing model
8. Validate predictions using an Actual vs Predicted plot
9. Save the selected model

### Model Comparison

| Model | RMSE | R² Score |
|---|---:|---:|
| Linear Regression | 0.746 | 0.576 |
| Ridge Regression | 0.746 | 0.576 |
| Decision Tree | 0.724 | 0.600 |

### Final Model

The Decision Tree Regressor was selected based on the lowest RMSE among
the evaluated models.

### Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Google Colab

### Files

- `AI_ML_Task2_Model_Comparison.ipynb` – Task 2 Jupyter/Colab notebook
- `AI_ML_Task2_Model_Comparison_Report.pdf` – Task 2 report
- `best_california_housing_model.pkl` – Saved trained model
