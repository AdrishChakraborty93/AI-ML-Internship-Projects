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

## Task 3: Model Validation, Overfitting Control & Hyperparameter Tuning

### Objective

Evaluate model generalization, identify overfitting, apply cross-validation, and tune Decision Tree hyperparameters using GridSearchCV.

### Dataset

California Housing Dataset

### Techniques Used

* Train-test split
* Training and testing RMSE comparison
* 5-fold cross-validation
* GridSearchCV for hyperparameter tuning
* RMSE and R² evaluation
* Comparison of Linear Regression, Ridge Regression, and a tuned Decision Tree

### Best Hyperparameters

* `max_depth`: 10
* `min_samples_split`: 10

### Results

| Model               | Test RMSE | Test R² |
| ------------------- | --------: | ------: |
| Tuned Decision Tree |     0.645 |   0.682 |
| Ridge Regression    |     0.746 |   0.576 |
| Linear Regression   |     0.746 |   0.576 |

The default Decision Tree achieved an almost-zero training RMSE, indicating severe overfitting. GridSearchCV identified a better-performing configuration, and the tuned Decision Tree achieved the lowest test RMSE among the three models.

### Key Learning

This task demonstrated why training performance alone is insufficient, and how cross-validation and hyperparameter tuning can help select a model that generalizes better.

### Files

- `AI_ML_Task3_Model_Validation_Tuning.ipynb` – Task 3 Jupyter/Colab notebook
- `AI_ML_Task3_Model_Validation_Overfitting_Tuning_Report.pdf` – Task 3 report

## Task 4: Classification Models, Evaluation Metrics & Handling Imbalanced Data

### Objective

Compare classification models on the scikit-learn Breast Cancer
Wisconsin dataset, evaluate confusion matrices and classification
metrics, examine ROC-AUC, and test class weighting.

### Dataset

-   Dataset: `sklearn.datasets.load_breast_cancer`
-   Samples/features: 569 samples, 30 features
-   Labels: malignant = `0`; benign = `1`
-   Class distribution: 212 malignant, 357 benign
-   Split: 80/20, `random_state=42`, stratified; test set has 114 cases
    (42 malignant, 72 benign)
-   Scaling: `StandardScaler` fitted on training data only for Logistic
    Regression; Decision Tree used unscaled features.

### Model comparison

  -----------------------------------------------------------------------------
  Model            Accuracy    Malignant    Malignant Malignant F1      ROC-AUC
                               Precision       Recall                  (Benign)
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic            0.982        0.976        0.976        0.976        0.995
  Regression                                                       

  Balanced            0.956        0.911        0.976        0.943        0.995
  Logistic                                                         
  Regression                                                       

  Decision            0.912        0.848        0.929        0.886        0.916
  Tree                                                             
  -----------------------------------------------------------------------------

### Confusion matrices

Rows are actual classes and columns are predicted classes, in this
order: malignant, benign.

### Logistic Regression

`[[41, 1], [1, 71]]`

41 malignant and 71 benign cases were correctly classified. One
malignant case was predicted benign and one benign case was predicted
malignant.

### Balanced Logistic Regression

`[[41, 1], [4, 68]]`

41 malignant and 68 benign cases were correctly classified. One
malignant case was predicted benign and four benign cases were predicted
malignant.

### Decision Tree

`[[39, 3], [7, 65]]`

39 malignant and 65 benign cases were correctly classified. Three
malignant cases were predicted benign and seven benign cases were
predicted malignant.

### Interpretation

-   Logistic Regression performed best overall on this test split, with
    98.2% accuracy and 0.995 ROC-AUC for the benign class.
-   Balanced Logistic Regression retained malignant recall at 0.976, but
    malignant precision fell to 0.911 and accuracy to 0.956. In this
    run, it classified more benign cases as malignant.
-   Decision Tree had the lowest test performance among the three
    models, with 0.912 accuracy and 0.916 ROC-AUC for the benign class.
-   `predict_proba(X)[:, 1]` refers to the benign class because label 1
    is benign. Malignant precision, recall, and F1 are calculated
    separately for label 0.

### Files

- `AI_ML_Task4_Classification_Evaluation.ipynb` – Task 4 Jupyter/Colab notebook
- `AI_ML_Task4_Classification_Report.pdf` – Task 4 report



