import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

# Load dataset (replace with your food dataset)
data = load_wine()
X, y = data.data, data.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base classifier
base_clf = DecisionTreeClassifier(random_state=42)

# Bagging
bagging_clf = BaggingClassifier(base_estimator=base_clf, n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)

# Boosting (AdaBoost)
boosting_clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=50, random_state=42)
boosting_clf.fit(X_train, y_train)
boosting_pred = boosting_clf.predict(X_test)
boosting_acc = accuracy_score(y_test, boosting_pred)


# Stacking
estimators = [
    ('bagging', BaggingClassifier(base_estimator=base_clf, n_estimators=10, random_state=42)),
    ('boosting', AdaBoostClassifier(base_estimator=base_clf, n_estimators=10, random_state=42)),
    ('grad_boost', GradientBoostingClassifier(n_estimators=10, random_state=42))
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_pred)

# Plotting the performance
labels = ['Bagging', 'Boosting',  'Stacking']
accuracies = [bagging_acc, boosting_acc, stacking_acc]

plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color=['blue', 'green', 'purple'])
plt.xlabel('Ensemble Methods')
plt.ylabel('Accuracy')
plt.title('Performance Comparison of Ensemble Methods')
for i in range(len(labels)):
    plt.text(i, accuracies[i] + 0.01, f'{accuracies[i]:.4f}', ha='center')
plt.ylim(0, 1)
plt.show()
