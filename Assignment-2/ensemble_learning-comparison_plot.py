import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a range for n_estimators
n_estimators_range = list(range(1, 101, 10))

# Define classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
}

# Dictionary to hold accuracy results
accuracy_results = {name: [] for name in classifiers.keys()}

# Run experiments
for name, clf in classifiers.items():
    for n_estimators in n_estimators_range:
        if 'n_estimators' in clf.get_params().keys():
            clf.set_params(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[name].append(accuracy)
    if name == 'Decision Tree':  # Decision Tree doesn't use n_estimators, so repeat its score
        accuracy_results[name] = accuracy_results[name] * len(n_estimators_range)

# Plot results for ensemble methods
for name, accuracies in accuracy_results.items():
    if name != 'Decision Tree':  # Skip plotting for Decision Tree
        plt.plot(n_estimators_range, accuracies, marker='o', label=name)

# Add Decision Tree accuracy to the plot
dt_accuracy = accuracy_results['Decision Tree'][0]
plt.hlines(dt_accuracy, xmin=1, xmax=100, colors='r', linestyles='dashed', label='Decision Tree')

plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison')
plt.legend()
plt.show()
