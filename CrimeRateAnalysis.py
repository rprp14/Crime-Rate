import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------
# Load Dataset
# -----------------------------------------------
print("Loading dataset...")
try:
    crimerate = pd.read_csv("BaltimoreCrimerate.csv")
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'BaltimoreCrimerate.csv' not found. Please ensure the file exists in the specified path.\n")
    exit()

# Display the first few rows of the dataset
print("Dataset Preview (First 5 Rows):")
print(crimerate.head(), "\n")

# -----------------------------------------------
# Exploratory Data Analysis (EDA)
# -----------------------------------------------
print("Dataset Shape (Rows, Columns):", crimerate.shape, "\n")
print("Column Data Types:")
print(crimerate.dtypes, "\n")

# Check for missing values
print("Missing Values in Each Column:")
print(crimerate.isnull().sum(), "\n")

# Descriptive statistics for numerical columns
print("Descriptive Statistics for Numerical Columns:")
print(crimerate.describe(), "\n")

# Analyze unique values in key columns
if 'CrimeTime' in crimerate.columns:
    print("Value Counts for 'CrimeTime':")
    print(crimerate['CrimeTime'].value_counts(), "\n")

if 'Neighborhood' in crimerate.columns:
    print("Value Counts for 'Neighborhood':")
    print(crimerate['Neighborhood'].value_counts(), "\n")

if 'District' in crimerate.columns:
    print("Value Counts for 'District':")
    print(crimerate['District'].value_counts(), "\n")

if 'Weapon' in crimerate.columns:
    print("Value Counts for 'Weapon':")
    print(crimerate['Weapon'].value_counts(), "\n")

if 'Inside/Outside' in crimerate.columns:
    print("Value Counts for 'Inside/Outside':")
    print(crimerate['Inside/Outside'].value_counts(), "\n")

# -----------------------------------------------
# Data Preprocessing
# -----------------------------------------------
# Replace 'Inside/Outside' values with shorthand
if 'Inside/Outside' in crimerate.columns:
    crimerate['Inside/Outside'] = crimerate['Inside/Outside'].replace({'Inside': 'I', 'Outside': 'O'})

# Splitting 'CrimeDate' into Year, Month, and Day
if 'CrimeDate' in crimerate.columns:
    crimerate['CrimeYear'] = crimerate['CrimeDate'].str.split(pat='/').str[2].astype(int)
    crimerate['CrimeMonth'] = crimerate['CrimeDate'].str.split(pat='/').str[0].astype(int)
    crimerate['CrimeDay'] = crimerate['CrimeDate'].str.split(pat='/').str[1].astype(int)
    crimerate = crimerate.drop(['CrimeDate'], axis=1)

# Splitting 'CrimeTime' into Hours, Minutes, and Seconds
if 'CrimeTime' in crimerate.columns:
    crimerate['CrimeHours'] = crimerate['CrimeTime'].str.split(pat=':').str[0].astype(int)
    crimerate['CrimeMinutes'] = crimerate['CrimeTime'].str.split(pat=':').str[1].astype(int)
    crimerate['CrimeSeconds'] = crimerate['CrimeTime'].str.split(pat=':').str[2].astype(int)
    crimerate = crimerate.drop(['CrimeTime'], axis=1)

# Display updated dataset
print("Updated Dataset (First 5 Rows):")
print(crimerate.head(), "\n")

# -----------------------------------------------
# Data Visualization
# -----------------------------------------------

# Yearly Crime Trends
print("Visualizing Yearly Crime Trends...")
yearly_trend = crimerate.groupby('CrimeYear').size()
plt.figure(figsize=(10, 6))
yearly_trend.plot(kind='line', marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Yearly Crime Trends', fontsize=16)
plt.xlabel('Crime Year', fontsize=12)
plt.ylabel('Crime Count', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Monthly Crime Distribution
print("Visualizing Monthly Crime Distribution...")
monthly_trend = crimerate.groupby('CrimeMonth').size()
plt.figure(figsize=(10, 6))
monthly_trend.plot(kind='bar', color='skyblue', width=0.8)
plt.title('Monthly Crime Distribution', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Crime Count', fontsize=12)
plt.xticks(ticks=range(12), labels=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
], rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# District Crime Distribution
print("Visualizing Crime Distribution by District...")
if 'District' in crimerate.columns:
    district_crime = crimerate['District'].value_counts(normalize=True) * 100
    plt.figure(figsize=(12, 6))
    sns.barplot(x=district_crime.index, y=district_crime.values, palette='Blues_d')
    plt.title('Crime Distribution by District (%)', fontsize=16)
    plt.xlabel('District', fontsize=12)
    plt.ylabel('Percentage of Total Crimes', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# Predictive Modeling
# -----------------------------------------------
if 'CrimeCode' in crimerate.columns:
    print("Preparing Data for Predictive Modeling...\n")
    
    # Prepare features (X) and target (y)
    X = crimerate[['CrimeYear', 'CrimeMonth', 'CrimeDay', 'CrimeHours', 'District', 'Inside/Outside']]
    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical data
    y = crimerate['CrimeCode']

    # Split data into training and testing sets
    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!\n")

    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = model.predict(X_test)

    # Display classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred), "\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred), "\n")

else:
    print("Column 'CrimeCode' not found. Predictive modeling cannot proceed.\n")
