# Logistic Regression Analysis: Car Insurance Dataset
Machine learning is very popular in the insurance market, which has long been renowned for being data-driven.
Working on behalf of On the Road car insurance, you'll investigate their customer data to identify the feature that produces the most accurate logistic regression model for them to predict whether a claim will be made against a policy!

## Project Overview
This project aims to build a logistic regression model to predict whether a customer will make a claim on their car insurance during the policy period. The goal is to identify the single feature that results in the best-performing model, as measured by accuracy. The dataset provided (`car_insurance.csv`) contains various features about customers and their car insurance policies.


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

To ensure the dataset is complete for analysis since missing values can distort model results, reduce predictive accuracy, and cause errors during computations.
 - e.g. If credit_score is missing for many customers, it could distort risk assessments, as credit scores are often correlated with claim frequency or fraud likelihood.

Missing values in key columns like "credit\_score" or "annual\_mileage" might affect predictions. I fill them up using simple median imputation (since the missing values doesnt make up more than 20-30% of the features) and compare summary statitics afterwards

```python
# Check and Fill Missing Values
print("\nMissing Values:\n", cars.isnull().sum())
cars["credit_score"].fillna(cars["credit_score"].mean(), inplace=True)
cars["annual_mileage"].fillna(cars["annual_mileage"].mean(), inplace=True)
print("\nAfter Filling Missing Values:\n", cars.isnull().sum())
```
- `isnull().sum()`: Counts missing values for each column.
- `fillna(mean)`: Fills missing values with the column's mean, preserving the dataset's overall distribution.

```python
# Summary statistics
print("Before Imputation:")
print(cars[["credit_score", "annual_mileage"]].describe())

print("\nAfter Imputation:")
print(cars[["credit_score", "annual_mileage"]].describe())
```
```
Before Imputation:
       credit_score  annual_mileage
count  10000.000000    10000.000000
mean       0.516718    11726.000000
std        0.130781     2681.649329
min        0.053358     2000.000000
25%        0.431509    10000.000000
50%        0.525033    12000.000000
75%        0.607607    13000.000000
max        0.960819    22000.000000

After Imputation:
       credit_score  annual_mileage
count  10000.000000    10000.000000
mean       0.516718    11726.000000
std        0.130781     2681.649329
min        0.053358     2000.000000
25%        0.431509    10000.000000
50%        0.525033    12000.000000
75%        0.607607    13000.000000
max        0.960819    22000.000000
```
## Visualizing the Distribution of Target Variable

To understand the balance of the target variable (e.g., the class distribution of "outcome").

Outcome Counts:
- No Claims (0): 7,000 instances
- Claims (1): 3,000 instances

- Given the moderately imbalance data on the outcome counts, we'll evaluate our model using multiple metrics beyond accuracy, such as precision, recall, and F1-score to provide a more comprehensive view of model performance, especially in identifying the minority class (claims)

```python
sns.countplot(x="outcome", data=cars)
plt.title("Distribution of Outcome Variable")
plt.show()
```

![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/c40936bac8cd3a3331994eec83fa0875cb680edb/Images/Distribution%20of%20Outcome%20Variable.png)

## Preprocessing and Feature Selection

To have a clear understanding of the data (e.g. variability and distribution of categories) before deciding which features to use
```
# Inspect unique values for categorical features
for col in cars.columns:
    if cars[col].dtype == 'object':
        print(f"{col} unique values:\n{cars[col].value_counts()}\n")

# Inspect unique values for numerical features
for col in cars.columns:
    if cars[col].dtype != 'object':
        print(f"{col} unique values:\n{cars[col].value_counts()}\n")
```

### Dropping Irrelevant Columns

To remove non-predictive features, such as IDs, which do not contain meaningful information for the model.

```python
irrelevant_columns = ["id", "outcome"]
features = cars.drop(columns=irrelevant_columns).columns
```
- `drop(columns)`: Removes specified columns from the dataset.
  
## Feature Evaluation Using Logistic Regression

To evaluate the predictive power of each feature in the dataset with respect to the target variable (outcome) and identify the best predictor for insurance claims.

### Process

#### Logistic Regression Model
- A logistic regression model was created for each feature to predict the target variable (outcome).
- The model output was a probability score for each observation, indicating the likelihood of a positive outcome (1, a claim).

#### Threshold for Classification
- The probability scores were converted into binary predictions (0 or 1) using a threshold of 0.5.

#### Evaluation Metrics
The following metrics were calculated to assess the model performance for each feature:
- **Accuracy**: Proportion of correct predictions out of the total predictions.
- **Precision**: Proportion of predicted claims (1) that are actual claims.
- **Recall**: Proportion of actual claims (1) correctly predicted by the model.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure of both metrics.

#### Sorting by F1-Score
- The features were ranked based on their F1-Scores. This ensures that the evaluation prioritizes features that balance precision and recall, which is crucial for imbalanced datasets.


```python
# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from statsmodels.formula.api import logit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# List to store the evaluation results and models
evaluation_results = []
models = []

# Loop through each feature
for col in features:
    # Train logistic regression model for the single feature
    model = logit(f"outcome ~ {col}", data=cars).fit(disp=False)
    models.append(model)  # Store the model
    
    # Predict probabilities and convert to binary using a threshold
    y_pred_prob = model.predict(cars[[col]])  # Ensure correct format
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    # Evaluate model metrics
    accuracy = accuracy_score(cars["outcome"], y_pred_class)
    precision = precision_score(cars["outcome"], y_pred_class)
    recall = recall_score(cars["outcome"], y_pred_class)
    f1 = f1_score(cars["outcome"], y_pred_class)
    
    # Store the results
    evaluation_results.append({
        "Feature": col,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

# Convert results to a DataFrame for visualization
evaluation_df = pd.DataFrame(evaluation_results)

# Sort by F1-Score
evaluation_df = evaluation_df.sort_values(by="F1-Score", ascending=False)

# Display the evaluation results
print(evaluation_df)

# Retrieve the best feature based on F1-Score
best_feature_row = evaluation_df.iloc[0]  # First row of sorted DataFrame
best_feature_name = best_feature_row["Feature"]
best_accuracy = best_feature_row["Accuracy"]

```
## Outcome
```
               Feature  Accuracy  Precision    Recall  F1-Score
2    driving_experience    0.7771   0.628045  0.707628  0.665466
6     vehicle_ownership    0.7351   0.579868  0.560804  0.570177
0                   age    0.7747   0.718254  0.462177  0.562439
4                income    0.7425   0.653804  0.378551  0.479483
5          credit_score    0.7053   0.572430  0.234599  0.332805
11       annual_mileage    0.6904   0.609467  0.032876  0.062386
1                gender    0.6867   0.000000  0.000000  0.000000
3             education    0.6867   0.000000  0.000000  0.000000
7          vehicle_year    0.6867   0.000000  0.000000  0.000000
8               married    0.6867   0.000000  0.000000  0.000000
9              children    0.6867   0.000000  0.000000  0.000000
10          postal_code    0.6867   0.000000  0.000000  0.000000
12         vehicle_type    0.6867   0.000000  0.000000  0.000000
13  speeding_violations    0.6867   0.000000  0.000000  0.000000
14                 duis    0.6867   0.000000  0.000000  0.000000
15       past_accidents    0.6867   0.000000  0.000000  0.000000
```
- The results showed each feature's performance across the defined metrics.
- The features were sorted by F1-Score to identify the best predictor for claims (1).
- This approach ensured that the selected feature is effective in identifying claims while maintaining a balance between minimizing false positives and false negatives.

### **Conclusion:**
By analyzing various features, we identified that driving_experience is the best single predictor of whether a customer will make a claim on their car insurance during the policy period. This feature achieved the highest model accuracy of 77.71%. The logistic regression model shows that as driving experience increases, the likelihood of making a claim decreases.
### Visualizing Performance of the Best Feature

**Purpose:** To evaluate the model's performance through a confusion matrix and detailed classification metrics.

**Why:** Metrics like precision, recall, and F1-score show how well the model predicts each class, helping identify strengths and weaknesses.

```python
# Train the best model again to ensure everything is correctly linked
best_model = logit(f"outcome ~ {best_feature_name}", data=cars).fit(disp=False)

# Visualize Performance of the Best Feature
y_pred = best_model.predict(cars[[best_feature_name]])
y_pred_class = (y_pred >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(cars["outcome"], y_pred_class)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Best Feature Model")
plt.show()

# Performance Metrics
print("Classification Report:\n", classification_report(cars["outcome"], y_pred_class))
print("Accuracy Score:", accuracy_score(cars["outcome"], y_pred_class))

```
```
Classification Report:
               precision    recall  f1-score   support

         0.0       0.86      0.81      0.83      6867
         1.0       0.63      0.71      0.67      3133

    accuracy                           0.78     10000
   macro avg       0.74      0.76      0.75     10000
weighted avg       0.79      0.78      0.78     10000

Accuracy Score: 0.7771

```
- `confusion_matrix`: Summarizes model predictions versus actuals.
- `classification_report`: Provides detailed metrics for precision, recall, and F1-score.
![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/ae36cb342ea0c9f8aadb9d37ec2f3d2614c51dcc/Confusion%20Matrix.png)
## Summary

We built a model to predict whether a customer will make a claim on their car insurance during the policy period. Our goal was to find the single best feature that could give us the most accurate predictions.

### Key Findings

- **Best Predictor**: The most important factor in predicting whether a customer will make a claim is their **driving experience**.
  - **Insight**: Customers with more driving experience are less likely to make a claim.

- **Model Accuracy**: Our model correctly predicts whether a customer will make a claim **77.71%** of the time.
  - **Insight**: This means that out of every 100 predictions, about 78 will be correct.

- **Precision**: When our model predicts that a customer will make a claim, it is correct **62.80%** of the time.
  - **Insight**: This means that out of every 10 claims predicted by our model, about 6 will actually be claims. The remaining 4 will be false alarms.

- **Recall**: Our model identifies **70.77%** of the actual claims.
  - **Insight**: This means that out of every 10 actual claims, our model correctly predicts about 7. The remaining 3 are missed by our model.

- **Balanced Performance**: The **F1-Score** of **66.50%** indicates a good balance between precision (accuracy of predictions) and recall (ability to identify actual claims).
  - **Insight**: Our model strikes a balance between making accurate predictions and identifying as many claims as possible.

### Visualizing Model Performance

We used a confusion matrix to visualize the performance of our model. Here’s a simplified explanation:

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 5554         | 1313          |
| **Actual Yes** | 916          | 2217          |

- **True Positives**: The model correctly identified 2,217 cases where customers made a claim.
- **True Negatives**: The model correctly identified 5,554 cases where customers did not make a claim.
- **False Positives**: The model incorrectly predicted 1,313 cases as claims when they were not.
- **False Negatives**: The model missed 916 actual claims, predicting them as non-claims.

```

# Visualize Feature Importance for All Features
odds_ratios = pd.DataFrame({
    feature: np.exp(models[i].params) for i, feature in enumerate(features)
})

# Calculate the mean odds ratios for each feature
mean_odds_ratios = odds_ratios.mean(axis=1).sort_values(ascending=False)

# Plot the feature importance as a horizontal bar chart
mean_odds_ratios.plot(kind="barh", title="Feature Importance Based on Odds Ratios", color="teal")
plt.xlabel("Odds Ratio")
plt.ylabel("Features")
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Driving Experience Analysis
print(cars['driving_experience'].value_counts())
sns.countplot(x='driving_experience', hue='outcome', data=cars)
plt.title("Outcome Distribution by Driving Experience")
plt.show()

```
![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/32cdd57927337d998f2cb8257e201c3a60860f33/feature%20importance%20based%20on%20odd%20ratios.png)
![](https://github.com/snoowbirvd/Logistic-Regression-Analysis-Car-Insurance-Dataset/blob/9ed4cb74cb9d3073f36867f1a4e88df05bfd7ed5/Images/Outcome%20Distribution%20by%20Best%20Feature.png)


