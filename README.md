# Student Performance Classification

A machine learning project designed to classify student performance based on academic, demographic, and behavioural attributes. This repository includes a complete workflow: Exploratory Data Analysis (EDA), data preprocessing, model building, evaluation, and insights from feature importance.

---

## Repository Contents

* **Student_performance_Classification.ipynb** – Jupyter Notebook containing:

  * EDA
  * Data preprocessing
  * Model training & evaluation
  * Feature importance analysis
* **Student_performance_data_.csv** – Dataset used for model development.
* **PDF Report** – A concise 4 page report summarizing the project.

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

* Decision Tree Classifier
* Random Forest Classifier

**Best Model:** Random Forest

* Captures non-linear patterns
* Provides interpretable feature importance

---

## Model Performance Summary

Evaluation metrics used:

* Confusion Matrix
* F1-score

The Random Forest Classifier outperformed Decision Tree model due to stronger generalization and better handling of complex feature interactions.

---

## Feature Importance Insights

Top contributing features:

* Previous exam performance
* Attendance percentage
* Study hours
* Parental support
* Tutoring

These features had the strongest influence on the model's predictions.



## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/Mayurroro/Student-Performance-Classification
   ```

2. Open the notebook:

   ```bash
   jupyter notebook Student_performance_Classification.ipynb
   ```
3. Run all cells to reproduce results.

4. Change Data Set's absolute path to your local path.
---
This project was created as part of a machine learning learning module focusing on classification techniques, model evaluation, and data-driven insights.
