# Logistic Regression Analysis: Car Insurance Dataset

Dataset: [car_insurance.csv](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/77b3a79ae2d4307a019ea364117b0dcd6266eddd/car_insurance.csv)

![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/70ad21d14253d304f4c3dcace7e5e272c9b95f37/Images/DataCamp%20Project%20Details.png)


# Holistic Approach:
## Inspecting the Data

```python
import pandas as pd
from statsmodels.formula.api import logit
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load data
cars = pd.read_csv("car_insurance.csv")

# Data Exploration
print(cars.info())
print(cars.head())
print(cars.describe(include='all'))
```
- `pd.read_csv("car_insurance.csv")`: Loads the dataset into a DataFrame.
- `info()`: Provides an overview of the dataset, including column names, non-null counts, and data types.
- `head()`: Displays the first few rows of the dataset.
- `describe(include='all')`: Summarizes statistics for all columns, including counts, means, and unique values.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 18 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   id                   10000 non-null  int64  
 1   age                  10000 non-null  int64  
 2   gender               10000 non-null  int64  
 3   driving_experience   10000 non-null  object 
 4   education            10000 non-null  object 
 5   income               10000 non-null  object 
 6   credit_score         9018 non-null   float64
 7   vehicle_ownership    10000 non-null  float64
 8   vehicle_year         10000 non-null  object 
 9   married              10000 non-null  float64
 10  children             10000 non-null  float64
 11  postal_code          10000 non-null  int64  
 12  annual_mileage       9043 non-null   float64
 13  vehicle_type         10000 non-null  object 
 14  speeding_violations  10000 non-null  int64  
 15  duis                 10000 non-null  int64  
 16  past_accidents       10000 non-null  int64  
 17  outcome              10000 non-null  float64
dtypes: float64(6), int64(7), object(5)
memory usage: 1.4+ MB
None
       id  age  gender  ... duis past_accidents outcome
0  569520    3       0  ...    0              0     0.0
1  750365    0       1  ...    0              0     1.0
2  199901    0       0  ...    0              0     0.0
3  478866    0       1  ...    0              0     0.0
4  731664    1       1  ...    0              1     1.0

[5 rows x 18 columns]
                   id           age  ...  past_accidents       outcome
count    10000.000000  10000.000000  ...    10000.000000  10000.000000
unique            NaN           NaN  ...             NaN           NaN
top               NaN           NaN  ...             NaN           NaN
freq              NaN           NaN  ...             NaN           NaN
mean    500521.906800      1.489500  ...        1.056300      0.313300
std     290030.768758      1.025278  ...        1.652454      0.463858
min        101.000000      0.000000  ...        0.000000      0.000000
25%     249638.500000      1.000000  ...        0.000000      0.000000
50%     501777.000000      1.000000  ...        0.000000      0.000000
75%     753974.500000      2.000000  ...        2.000000      1.000000
max     999976.000000      3.000000  ...       15.000000      1.000000

[11 rows x 18 columns]
```
### Handle Missing Values

**Purpose:** To ensure the dataset is complete for analysis. Missing values can distort model results, reduce predictive accuracy, and cause errors during computations.

**Why:** Missing values in key columns like "credit\_score" or "annual\_mileage" might affect predictions. We fill them using the mean because it is a simple imputation method that avoids introducing bias and retains numerical consistency.

```python
# Check and Fill Missing Values
print("\nMissing Values:\n", cars.isnull().sum())
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)
print("\nAfter Filling Missing Values:\n", cars.isnull().sum())
```
- `isnull().sum()`: Counts missing values for each column.
- `fillna(mean)`: Fills missing values with the column's mean, preserving the dataset's overall distribution.

## Visualizing the Distribution of Target Variable

Purpose: To understand the balance of the target variable (e.g., the class distribution of "outcome").

Why: Imbalanced classes can skew the model toward predicting the majority class, leading to poor performance on the minority class. By visualizing the distribution, we can assess whether techniques like oversampling, undersampling, or weighted modeling are needed to handle class imbalance.\

Findings:
After visualizing the target variable, the plot shows that the "outcome" is balanced (both classes have a roughly equal number of observations). This suggests that we do not need to apply class balancing techniques at this stage, as the data distribution supports unbiased training.

```python
sns.countplot(x="outcome", data=cars)
plt.title("Distribution of Outcome Variable")
plt.show()
```

![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/c40936bac8cd3a3331994eec83fa0875cb680edb/Images/Distribution%20of%20Outcome%20Variable.png)

### Correlation Heatmap

**Purpose:** To identify relationships between numerical variables and check for multicollinearity.

**Why:** High correlations between features can cause redundancy, making the model less interpretable and possibly unstable. Visualizing correlations guides us in removing or combining features.

```python
numerical_cols = cars.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 6))
sns.heatmap(cars[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```
- `select_dtypes`: Selects numerical columns for correlation analysis.
- `heatmap`: Visualizes the correlation matrix.
  
## Preprocessing and Feature Selection

### Dropping Irrelevant Columns

**Purpose:** To remove non-predictive features, such as IDs, which do not contain meaningful information for the model.

**Why:** Columns like "id" don't influence the target variable and only add unnecessary noise. Removing them ensures cleaner and more efficient modeling.

```python
irrelevant_columns = ["id", "outcome"]
features = cars.drop(columns=irrelevant_columns).columns
```
- `drop(columns)`: Removes specified columns from the dataset.
  
### Analyzing Categorical Features

**Purpose:** To evaluate variability and understand the distribution of categories in non-numerical features.

**Why:** Features with little variability contribute minimally to predictions. Understanding category distributions helps refine feature selection.

```python
for col in features:
    if cars[col].dtype == 'object':
        print(f"{col} unique values:\n{cars[col].value_counts()}\n")
```
- `value_counts()`: Displays the count of each category.
- Iterates through object-type columns to inspect their distributions.
  
## Logistic Regression Analysis

### Iterative Model Building and Evaluation

**Purpose:** To assess the predictive power of each feature by training individual logistic regression models.

**Why:** Building a model for each feature separately highlights their importance, simplifying initial model exploration. Odds ratios provide interpretable insights about feature impacts.

```python
models = []
accuracies = []
summary_data = []

for col in features:
    model = logit(f"outcome ~ {col}", data=cars).fit(disp=False)
    models.append(model)

    # Calculate Accuracy
    conf_matrix = model.pred_table()
    tn, fp, fn, tp = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]
    acc = (tn + tp) / (tn + fp + fn + tp)
    accuracies.append(acc)

    # Collect Summary Data
    summary_data.append({
        "Feature": col,
        "Odds Ratios": np.exp(model.params),
        "Accuracy": acc
    })

    # Print Detailed Model Summary
    print(f"Feature: {col}")
    print(model.summary())
    print("\nOdds Ratios:\n", np.exp(model.params))
    print("-" * 50)
```
- `logit`: Builds logistic regression models for binary outcomes.
- `params`: Extracts feature coefficients for odds ratio calculation.
- `pred_table`: Generates a confusion matrix for accuracy computation.

### Identifying the Best Feature

**Purpose:** To select the most predictive feature for further analysis and refined modeling.

**Why:** Focusing on the best feature reduces complexity and prioritizes performance. This is particularly useful for interpretable and practical applications.

```python
# Find the best feature
best_feature_index = accuracies.index(max(accuracies))
best_feature_name = features[best_feature_index]
best_accuracy = max(accuracies)

# Create and display the best feature DataFrame
best_feature_df = pd.DataFrame({
    "best_feature": [best_feature_name],
    "best_accuracy": [best_accuracy]
})
print("\nBest Feature and Accuracy:\n", best_feature_df)

best_model = models[best_feature_index]

print(f"\nBest Feature: {best_feature_name} with Accuracy: {best_accuracy:.2f}")
print("\nBest Feature Model Summary:\n")
print(best_model.summary())
```
```
Best Feature and Accuracy:
          best_feature  best_accuracy
0  driving_experience         0.7771

Best Feature: driving_experience with Accuracy: 0.78

Best Feature Model Summary:

                           Logit Regression Results                           
==============================================================================
Dep. Variable:                outcome   No. Observations:                10000
Model:                          Logit   Df Residuals:                     9996
Method:                           MLE   Df Model:                            3
Date:                Wed, 01 Jan 2025   Pseudo R-squ.:                  0.2487
Time:                        05:57:11   Log-Likelihood:                -4670.9
converged:                       True   LL-Null:                       -6217.2
Covariance Type:            nonrobust   LLR p-value:                     0.000
================================================================================================
                                   coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------
Intercept                        0.5238      0.035     15.043      0.000       0.456       0.592
driving_experience[T.10-19y]    -1.6844      0.054    -31.380      0.000      -1.790      -1.579
driving_experience[T.20-29y]    -3.4384      0.104    -32.957      0.000      -3.643      -3.234
driving_experience[T.30y+]      -4.4674      0.228    -19.557      0.000      -4.915      -4.020
================================================================================================
```
### Visualizing Performance of the Best Feature

**Purpose:** To evaluate the model's performance through a confusion matrix and detailed classification metrics.

**Why:** Metrics like precision, recall, and F1-score show how well the model predicts each class, helping identify strengths and weaknesses.

```python
y_pred = best_model.predict(cars[best_feature_name])
y_pred_class = (y_pred >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(cars["outcome"], y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Best Feature Model")
plt.show()

# Performance Metrics
print

```
- `confusion_matrix`: Summarizes model predictions versus actuals.
- `classification_report`: Provides detailed metrics for precision, recall, and F1-score.
  
```
0-9y      3530
10-19y    3299
20-29y    2119
30y+      1052
Name: driving_experience, dtype: int64
```
![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/ae36cb342ea0c9f8aadb9d37ec2f3d2614c51dcc/Confusion%20Matrix.png)
![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/9ed4cb74cb9d3073f36867f1a4e88df05bfd7ed5/Images/Outcome%20Distribution%20by%20Best%20Feature.png)
