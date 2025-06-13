# Boston House Price Prediction

This repository contains a Python project, designed for Google Colab, that builds and evaluates regression models to predict house prices based on various property and neighborhood features.

## 1. Task Objective
The primary objective of this project is to apply regression techniques to predict the median value of homes in various Boston suburbs. The project demonstrates a complete machine learning workflow, including data loading, preprocessing, exploratory data analysis, model training, and performance evaluation.

## 2. Dataset Used
The project utilizes the classic **Boston Housing dataset**, provided in the `BostonHousing.csv` file.

*   **Source:** Originally from the StatLib library maintained at Carnegie Mellon University.
*   **Instances:** 506 samples.
*   **Features:** The dataset consists of 14 attributes, including:
    *   `crim`: Per capita crime rate by town.
    *   `rm`: Average number of rooms per dwelling.
    *   `lstat`: Percentage of the population with a "lower status".
    *   `age`: Proportion of owner-occupied units built prior to 1940.
    *   `price`: The target variable, representing the median value of owner-occupied homes (in $1000s). We renamed this from the original `medv`.

## 3. Models and Methodology

The project follows a standard regression modeling pipeline:

1.  **Data Preprocessing:** The dataset was loaded and inspected. It was found to be clean with no missing values. The target column `medv` was renamed to `price` for clarity.

2.  **Exploratory Data Analysis (EDA):** A correlation heatmap was generated to understand the relationships between features. The analysis revealed that `rm` (number of rooms) has a strong positive correlation with price, while `lstat` (% lower status) has a strong negative correlation.

3.  **Model Preparation:**
    *   The data was split into features (X) and a target variable (y).
    *   Features were scaled using `StandardScaler` to ensure that all variables were on a comparable scale, which is important for linear models.
    *   The dataset was split into an 80% training set and a 20% testing set.

4.  **Models Applied:**
    Two different regression models were trained to predict house prices:
    *   **Linear Regression:** A fundamental linear model that establishes a simple relationship between features and the target.
    *   **Gradient Boosting Regressor:** A powerful ensemble model that builds multiple decision trees sequentially, with each tree correcting the errors of the previous one. It is generally more accurate than a single linear model.

## 4. Key Results and Findings

### Model Performance
The models were evaluated on the unseen test set using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

*   **Linear Regression:**
    *   **MAE:** ~$3.19k
    *   **RMSE:** ~$4.93k
*   **Gradient Boosting Regressor:**
    *   **MAE:** ~$2.07k
    *   **RMSE:** ~$2.91k

### Conclusion
The **Gradient Boosting Regressor significantly outperformed the Linear Regression model**, with substantially lower MAE and RMSE values. This indicates that the more complex, non-linear relationships captured by the Gradient Boosting model were better suited for this dataset.

The visualization of "Actual vs. Predicted Prices" further confirms this finding, showing that the Gradient Boosting predictions are more tightly clustered around the perfect-fit line compared to the Linear Regression predictions. An average prediction error of ~$2,070 (MAE for Gradient Boosting) demonstrates a strong predictive capability for this classic dataset.
