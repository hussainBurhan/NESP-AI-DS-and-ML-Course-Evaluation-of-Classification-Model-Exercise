# Heart Disease Prediction with Random Forest Classifier
## Overview
This project aims to predict the presence of heart disease using a Random Forest Classifier. The dataset used for this task contains various medical attributes that can be used to make predictions.

## Learning Outcomes
1. Utilizing pandas for data manipulation and preprocessing.
2. Implementing a Random Forest Classifier for binary classification tasks.
3. Evaluating model performance using various metrics such as accuracy, precision, recall, and ROC AUC.
4. Visualizing model performance with ROC curves and confusion matrices.
5. Understanding the importance of feature selection in machine learning model.

## Installation
Clone the repository: git clone https://github.com/your_username/heart-disease-prediction.git
Install the required packages: pandas numpy scikit-learn seaborn matplotlib
## Usage
Ensure you have Python and pip installed on your system.
Install the required packages
Run the main Python script main.py. The script will perform the following tasks:
  Load and preprocess the dataset.
  Train a Random Forest Classifier on the data.
  Evaluate the model's performance using cross-validation, ROC curve, confusion matrix, and classification report.
  Display various evaluation metrics including accuracy, precision, recall, and F1 score.
## Dependencies
Python 3.x
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
## File Descriptions
heart.csv: Dataset containing medical attributes and target labels.
main.py: Python script for training the Random Forest Classifier and evaluating the model.
## Output Example
Accuracy: 85.36%
Precision: 85.47%
Recall: 89.55%
F1 Score: 87.46%
Roc area under the curve: 0.91
Confusion Matrix:
[[23  5]
 [ 3 30]]
Classification Report:
              precision    recall  f1-score   support
           0       0.88      0.82      0.85        28
           1       0.86      0.91      0.88        33
    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.87      0.87      0.87        61


## Acknowledgments:
This program was created as part of the AI, DS and ML course offered by the National Emerging Skills Program (NESP).


