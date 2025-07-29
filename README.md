# California Housing Price Prediction using XGBoost Regressor

This project provides a comprehensive analysis and prediction of housing prices in California using a dataset that includes various features related to demographics, location, and housing characteristics. The primary objective is to build a robust regression model to accurately predict the median house value for a given district.

The notebook covers the entire machine learning workflow, including:
-   In-depth Exploratory Data Analysis (EDA)
-   Data cleaning and preprocessing (handling missing values, outliers, and categorical data)
-   Evaluation of multiple regression models
-   Hyperparameter tuning of the best-performing model (XGBoost)
-   Final model evaluation and conclusion.

## Dataset

The dataset used is the "California Housing Prices" dataset, containing data drawn from the 1990 California census. Each row represents a block group, which is the smallest geographical unit for which the U.S. Census Bureau publishes sample data.
[Kaggle](https://www.kaggle.com/code/emirhanhasrc/eda-multi-regressor-models-xgboost-regressor?scriptVersionId=253123870)
### Dataset Features

The dataset consists of 20,640 instances and 10 features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| **longitude** | A measure of how far west a house is; a higher value is farther west. | Numeric |
| **latitude** | A measure of how far north a house is; a higher value is farther north. | Numeric |
| **housing_median_age**| Median age of a house within a block; a lower number is a newer building. | Numeric |
| **total_rooms** | Total number of rooms within a block. | Numeric |
| **total_bedrooms** | Total number of bedrooms within a block. | Numeric |
| **population** | Total number of people residing within a block. | Numeric |
| **households** | Total number of households, a group of people residing within a home unit, for a block. | Numeric |
| **median_income** | Median income for households within a block of houses (measured in tens of thousands of US Dollars). | Numeric |
| **median_house_value**| **(Target)** Median house value for households within a block (measured in US Dollars). | Numeric |
| **ocean_proximity** | Location of the house with respect to the ocean/sea. | Categorical |

## Exploratory Data Analysis (EDA)

A detailed EDA was performed to understand the data's structure, distributions, and relationships.

1.  **Initial Data Inspection**:
    -   The dataset has **20,640 rows** and **10 columns**.
    -   The `total_bedrooms` column was identified as having **207 missing values**.
    -   The `ocean_proximity` column is the only categorical feature.

2.  **Outlier Detection and Handling**:
    -   The Interquartile Range (IQR) method was used to detect outliers in all numerical columns.
    -   Features like `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, and `median_house_value` were found to have a significant number of outliers.
    -   To preserve as much data as possible while ensuring the target variable's integrity, only the outliers from the `median_house_value` column were removed. This reduced the dataset from **20,640 to 19,569 instances**.
    -   After outlier removal, the remaining **200 missing values** in `total_bedrooms` were imputed using the median of that column.

3.  **Feature Analysis and Visualization**:
    -   **Distributions**: Histograms showed that most numerical features, particularly `total_rooms`, `population`, and `households`, are right-skewed. `median_income` is also skewed, with a long tail towards higher incomes.
    -   **Categorical Feature**: The `ocean_proximity` feature has five distinct categories, with `<1H OCEAN` being the most common, followed by `INLAND`. This feature was one-hot encoded for modeling.
    -   **Correlation Matrix**: A heatmap revealed a strong positive correlation of **0.69** between `median_income` and the target variable `median_house_value`. `total_rooms` and `households` also show a moderate positive correlation.

## Modeling and Evaluation

The preprocessed data was split into training (70%) and testing (30%) sets. A variety of regression models were evaluated to find the best performer.

### 1. Baseline Model Performance

Multiple regression models were trained on the training data and evaluated on the test set. The performance was measured using the **R² Score**.

| Model | R² Score (Test Set) |
| :--- | :--- |
| **XGBoost Regressor** | **0.8071** |
| **Random Forest Regressor** | 0.7935 |
| **Gradient Boost Regressor** | 0.7371 |
| **Ridge Regression** | 0.6264 |
| **Lasso Regression** | 0.6263 |
| **Linear Regression** | 0.6263 |
| **Decision Tree Regressor** | 0.5900 |
| **AdaBoost Regressor** | 0.3981 |
| **K-Neighbors Regressor** | 0.1521 |

The **XGBoost Regressor** provided the best baseline performance with an R² score of **80.71%**.

### 2. Hyperparameter Tuning

To optimize the XGBoost model, `RandomizedSearchCV` was used to explore a range of hyperparameters.

**Best Parameters Found:**
-   `n_estimators`: 300
-   `max_depth`: 8
-   `learning_rate`: 0.1
-   `colsample_bytree`: 1.0

### 3. Final Model Evaluation

The XGBoost model was re-trained using a slightly adjusted version of the best parameters to prevent overfitting. The final tuned model's performance on the training and test sets is shown below.

**XGBoost Regressor (Tuned) Performance:**

| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| **Root Mean Squared Error (RMSE)** | 24,640.35 | 41,418.26 |
| **Mean Absolute Error (MAE)** | 17,533.10 | 28,220.39 |
| **R² Score** | **0.9329** | **0.8144** |

## Conclusion

The exploratory data analysis revealed that `median_income` is the most significant predictor of `median_house_value`. After cleaning the data by handling outliers in the target variable and imputing missing values, a comparative analysis of nine different regression models was conducted.

The **XGBoost Regressor** emerged as the top-performing model. After hyperparameter tuning, the final model achieved an **R² score of 81.44%** on the test set. This indicates that the model can explain approximately **81.44%** of the variance in California housing prices, making it a strong and reliable predictor. The project successfully demonstrates a robust workflow for tackling a real-world regression problem.
