# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
np.random.seed(seed=1)

# Read the heart dataset
heart_csv = pd.read_csv('heart.csv')

# Separate features (x) and target (y)
x = heart_csv.drop('target', axis=1)
y = heart_csv['target']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize and train a RandomForestClassifier
clf = RandomForestClassifier().fit(x_train, y_train)

# Predict the target values
y_predicted = clf.predict(x_test)

# Cross-validation to evaluate the model
cv_score = np.mean(cross_val_score(clf, x, y, cv=5, scoring=None))
print(f'Accuracy : {cv_score * 100}%')
cv_score = np.mean(cross_val_score(clf, x, y, cv=5, scoring='precision'))
print(f'Precision : {cv_score * 100}%')
cv_score = np.mean(cross_val_score(clf, x, y, cv=5, scoring='recall'))
print(f'Recall : {cv_score * 100}%')
cv_score = np.mean(cross_val_score(clf, x, y, cv=5, scoring='f1'))
print(f'F1 score : {cv_score * 100}%')

# Calculate predicted probabilities for ROC curve
y_proba = clf.predict_proba(x_test)
y_proba_positive = y_proba[:,1]

# Generate ROC curve data
fpr, tpr, threshold = roc_curve(y_test, y_proba_positive)
print('FPR:')
print(fpr)
print('TPR:')
print(tpr)
print('Threshold:')
print(threshold)


# Define a function to plot the ROC curve
def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, color='red', label='AUC')
    plt.title('Area under the curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()

# Plot the ROC curve
plot_roc(fpr, tpr)

# Calculate and print ROC AUC score
print(f'Roc area under the curve: {roc_auc_score(y_test, y_proba_positive)}')

# Generate confusion matrix
conf_mat = confusion_matrix(y_test, y_predicted)

# Print and plot the confusion matrix
print(f'Confusion Matrix:')
print(conf_mat)

# Define a function to plot the confusion matrix
def plot_confmatrix(conf_mat):
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat, annot=True, cbar=False)
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.show()

# Plot the confusion matrix
plot_confmatrix(conf_mat)

# Generate and print classification report
class_repo = classification_report(y_test, y_predicted)
print(f'Classification Report:')
print(class_repo)

# Evaluate using individual metrics
print('Evaluation using functions:')
print('Accuracy Score:')
print(accuracy_score(y_test, y_predicted))
print('Precision Score:')
print(precision_score(y_test, y_predicted))
print('Recall Score:')
print(recall_score(y_test, y_predicted))
print('F1 Score:')
print(f1_score(y_test, y_predicted))
