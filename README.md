# Logistic Regression Analysis: Car Insurance Dataset

Dataset: [car_insurance.csv](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/77b3a79ae2d4307a019ea364117b0dcd6266eddd/car_insurance.csv)

![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/61e1175df7a3ea98e88d72b6fdc351338fd5af9f/Photos/DataCamp%20Project%20Details.png)


# Holistic Approach:
## Inspecting the Data

**Code Explanation:**

- `pd.read_csv("car_insurance.csv")`: Loads the dataset into a DataFrame.
- `info()`: Provides an overview of the dataset, including column names, non-null counts, and data types.
- `head()`: Displays the first few rows of the dataset.
- `describe(include='all')`: Summarizes statistics for all columns, including counts, means, and unique values.

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

### Handle Missing Values

**Purpose:** To ensure the dataset is complete for analysis. Missing values can distort model results, reduce predictive accuracy, and cause errors during computations.

**Why:** Missing values in key columns like "credit\_score" or "annual\_mileage" might affect predictions. We fill them using the mean because it is a simple imputation method that avoids introducing bias and retains numerical consistency.

**Code Explanation:**

- `isnull().sum()`: Counts missing values for each column.
- `fillna(mean)`: Fills missing values with the column's mean, preserving the dataset's overall distribution.

```python
# Check and Fill Missing Values
print("\nMissing Values:\n", cars.isnull().sum())
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)
print("\nAfter Filling Missing Values:\n", cars.isnull().sum())
```

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

### Correlation Heatmap

**Purpose:** To identify relationships between numerical variables and check for multicollinearity.

**Why:** High correlations between features can cause redundancy, making the model less interpretable and possibly unstable. Visualizing correlations guides us in removing or combining features.

**Code Explanation:**

- `select_dtypes`: Selects numerical columns for correlation analysis.
- `heatmap`: Visualizes the correlation matrix.

```python
numerical_cols = cars.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 6))
sns.heatmap(cars[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

## Preprocessing and Feature Selection

### Dropping Irrelevant Columns

**Purpose:** To remove non-predictive features, such as IDs, which do not contain meaningful information for the model.

**Why:** Columns like "id" don't influence the target variable and only add unnecessary noise. Removing them ensures cleaner and more efficient modeling.

**Code Explanation:**

- `drop(columns)`: Removes specified columns from the dataset.

```python
irrelevant_columns = ["id", "outcome"]
features = cars.drop(columns=irrelevant_columns).columns
```

### Analyzing Categorical Features

**Purpose:** To evaluate variability and understand the distribution of categories in non-numerical features.

**Why:** Features with little variability contribute minimally to predictions. Understanding category distributions helps refine feature selection.

**Code Explanation:**

- `value_counts()`: Displays the count of each category.
- Iterates through object-type columns to inspect their distributions.

```python
for col in features:
    if cars[col].dtype == 'object':
        print(f"{col} unique values:\n{cars[col].value_counts()}\n")
```

## Logistic Regression Analysis

### Iterative Model Building and Evaluation

**Purpose:** To assess the predictive power of each feature by training individual logistic regression models.

**Why:** Building a model for each feature separately highlights their importance, simplifying initial model exploration. Odds ratios provide interpretable insights about feature impacts.

**Code Explanation:**

- `logit`: Builds logistic regression models for binary outcomes.
- `params`: Extracts feature coefficients for odds ratio calculation.
- `pred_table`: Generates a confusion matrix for accuracy computation.

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

### Identifying the Best Feature

**Purpose:** To select the most predictive feature for further analysis and refined modeling.

**Why:** Focusing on the best feature reduces complexity and prioritizes performance. This is particularly useful for interpretable and practical applications.

**Code Explanation:**

- Identifies the feature with the highest accuracy.
- Displays its summary for detailed examination.

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

### Visualizing Performance of the Best Feature

**Purpose:** To evaluate the model's performance through a confusion matrix and detailed classification metrics.

**Why:** Metrics like precision, recall, and F1-score show how well the model predicts each class, helping identify strengths and weaknesses.

**Code Explanation:**

- `confusion_matrix`: Summarizes model predictions versus actuals.
- `classification_report`: Provides detailed metrics for precision, recall, and F1-score.

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
