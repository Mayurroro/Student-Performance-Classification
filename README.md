# Student Performance Classification

A machine learning project designed to classify student performance based on academic, demographic, and behavioural attributes. This repository includes a complete workflow: Exploratory Data Analysis (EDA), data preprocessing, model building, evaluation, and insights from feature importance.

---

## Repository Contents

* **Student_performance_Classification.ipynb** – Jupyter Notebook containing:

  * EDA
  * Data preprocessing
  * Model training & evaluation
  * Feature importance analysis
* **Student_performance_data_.csv** – Dataset used for model development
* **PDF Report** – A concise 2–3 page report summarizing the project (if added)

---

## Project Objective

The goal of this project is to build a classification model that predicts a student's performance category based on various input features. The project includes comparison of multiple ML models and insights from feature importance.

---

## Dataset Summary

The dataset contains information related to:

* Academic indicators (study hours, scores, attendance)
* Demographics (age, gender, parental education)
* Behavioral attributes

Basic preprocessing steps:

* Handling missing values
* Encoding categorical variables
* Normalizing numerical features
* Train-test split (80/20)

---

## Exploratory Data Analysis (EDA)

Key analyses performed:

* Distribution analysis of numerical & categorical variables
* Correlation heatmap
* Outlier detection
* Relationship patterns between features and performance

Insights:

* Study hours and attendance positively correlate with performance
* Previous exam scores are strong predictors

---

## Models Used

Multiple machine learning models were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)

**Best Model:** Random Forest

* Captures non-linear patterns
* Provides interpretable feature importance

---

## Model Performance Summary

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-score

The Random Forest Classifier outperformed other models due to stronger generalization and better handling of complex feature interactions.

---

## Feature Importance Insights

Top contributing features:

* Study hours
* Attendance percentage
* Previous exam performance
* Parental education level

These features had the strongest influence on the model's predictions.



## 🚀 How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Mayurroro/Student-Performance-Classification
   ```

2. Open the notebook:

   ```bash
   jupyter notebook Student_performance_Classification.ipynb
   ```
3. Run all cells to reproduce results.

---
This project was created as part of a machine learning learning module focusing on classification techniques, model evaluation, and data-driven insights.
