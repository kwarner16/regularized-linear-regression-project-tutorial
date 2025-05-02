import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

#TODO 1: Load and preprocess data
data = pd.read_csv("demographic_health_data.csv")

# Display basic information
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
data.info()
print("\nBasic Statistics:")
print(data.describe())

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Check for missing values
null_columns = data.columns[data.isnull().any()]
print("\nMissing Values:")
print(data[null_columns].isnull().sum())

# Check and remove duplicates
duplicates_all = data[data.duplicated(keep=False)]
print("\nDuplicates:")
print(duplicates_all)
data = data.drop_duplicates().reset_index(drop=True)

# Create target column
threshold = 13
data["diabetes_high"] = (data["diabetes_prevalence"] > threshold).astype(int)

# Select relevant columns
eda_cols = [
    "Obesity_prevalence",
    "Heart disease_prevalence",
    "COPD_prevalence",
    "PCTPOVALL_2018",
    "Percent of adults with a bachelor's degree or higher 2014-18",
    "Urban_rural_code",
    "diabetes_high",
]
cleaned_data = data[eda_cols]

#TODO 2: Visualization
# Distribution plots
fig, ax = plt.subplots(2, 4, figsize=(10, 7))
sns.histplot(ax=ax[0, 0], data=cleaned_data, x="Obesity_prevalence")
sns.boxplot(ax=ax[1, 0], data=cleaned_data, x="Obesity_prevalence")
sns.histplot(ax=ax[0, 1], data=cleaned_data, x="Heart disease_prevalence").set_ylabel(None)
sns.boxplot(ax=ax[1, 1], data=cleaned_data, x="Heart disease_prevalence").set_ylabel(None)
sns.histplot(ax=ax[0, 2], data=cleaned_data, x="COPD_prevalence").set_ylabel(None)
sns.boxplot(ax=ax[1, 2], data=cleaned_data, x="COPD_prevalence").set_ylabel(None)
sns.histplot(ax=ax[0, 3], data=cleaned_data, x="diabetes_high").set_ylabel(None)
sns.boxplot(ax=ax[1, 3], data=cleaned_data, x="diabetes_high").set_ylabel(None)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()

# Additional distribution plots
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
sns.histplot(ax=ax[0, 0], data=cleaned_data, x="PCTPOVALL_2018")
sns.boxplot(ax=ax[1, 0], data=cleaned_data, x="PCTPOVALL_2018")
sns.histplot(ax=ax[0, 1], data=cleaned_data, x="Percent of adults with a bachelor's degree or higher 2014-18").set_ylabel(None)
ax[0, 1].set_xlabel("% of Adults with Bachelor's/higher 2014-18")
sns.boxplot(ax=ax[1, 1], data=cleaned_data, x="Percent of adults with a bachelor's degree or higher 2014-18").set_ylabel(None)
sns.histplot(ax=ax[0, 2], data=cleaned_data, x="Urban_rural_code").set_ylabel(None)
sns.boxplot(ax=ax[1, 2], data=cleaned_data, x="Urban_rural_code").set_ylabel(None)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()

# Outlier analysis
def analyse_outliers(data, column: str):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]
    return outliers

print("\nOutlier Analysis:")
for col in cleaned_data:
    print(f"{col} outliers - {analyse_outliers(cleaned_data, col)}")

# Correlation analysis
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data=cleaned_data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#TODO 3: Prepare data for modeling
X = cleaned_data.drop("diabetes_high", axis=1)
y = cleaned_data["diabetes_high"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#TODO 4: Build and compare models
# Logistic Regression (baseline)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_pred = log_reg.predict(X_test_scaled)
log_accuracy = accuracy_score(y_test, log_pred)
log_roc_auc = roc_auc_score(y_test, log_pred)
print(f"\nBaseline Logistic Regression:")
print(f"Accuracy: {log_accuracy:.4f}")
print(f"ROC AUC: {log_roc_auc:.4f}")

# Lasso Logistic Regression with varying C (inverse of regularization strength)
C_values = np.logspace(-3, 1, 100)
accuracy_scores = []
roc_auc_scores = []

for C in C_values:
    lasso_log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42, max_iter=1000)
    lasso_log_reg.fit(X_train_scaled, y_train)
    lasso_pred = lasso_log_reg.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, lasso_pred))
    roc_auc_scores.append(roc_auc_score(y_test, lasso_pred))

# Plot metrics vs C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, accuracy_scores, label='Accuracy', marker='o')
plt.semilogx(C_values, roc_auc_scores, label='ROC AUC', marker='x')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Score')
plt.title('Lasso Logistic Regression: Accuracy and ROC AUC vs C')
plt.grid(True)
plt.legend()
plt.show()

#TODO 5: Optimize the model
# Grid search for best C
param_grid = {'C': np.logspace(-3, 1, 50)}
lasso_opt = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
grid_search = GridSearchCV(lasso_opt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_C = grid_search.best_params_['C']
best_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=42, max_iter=1000)
best_lasso.fit(X_train_scaled, y_train)
best_pred = best_lasso.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, best_pred)
best_roc_auc = roc_auc_score(y_test, best_pred)

print(f"\nOptimized Lasso Logistic Regression:")
print(f"Best C: {best_C:.4f}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"ROC AUC: {best_roc_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lasso.coef_[0]
})
feature_importance = feature_importance[feature_importance['Coefficient'] != 0].sort_values('Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title('Feature Importance in Optimized Lasso Model')
plt.show()

#TODO 6: Model comparison
print("\nModel Comparison:")
print(f"Baseline Logistic Regression - Accuracy: {log_accuracy:.4f}, ROC AUC: {log_roc_auc:.4f}")
print(f"Optimized Lasso Logistic Regression - Accuracy: {best_accuracy:.4f}, ROC AUC: {best_roc_auc:.4f}")