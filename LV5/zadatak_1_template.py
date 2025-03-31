import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# a) dio zadatka
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', marker='o', label="Train Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x', label="Test Data")

plt.legend()
plt.title("Train and Test Data by Class")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# b) dio zadatka
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c) dio zadatka
theta_1, theta_2 = LogRegression_model.coef_[0]
theta_0 = LogRegression_model.intercept_[0]

x1_values = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_values = - (theta_0 + theta_1 * x1_values) / theta_2

plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', marker='o', label="Train Data")
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', marker='x', label="Test Data")
plt.plot(x1_values, x2_values, 'k-', label="Decision Boundary")

plt.legend()
plt.title("Train and Test Data by Class")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# d) dio zadatka
y_test_pred = LogRegression_model.predict(X_test)
conf_matr = confusion_matrix(y_test, y_test_pred)
print(conf_matr)
display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
display.plot()
plt.show()

accuracy = (conf_matr[0][0] + conf_matr[1][1]) / (conf_matr[0][0] + conf_matr[0][1] + conf_matr[1][0] + conf_matr[1][1])
precision = (conf_matr[0][0]) / (conf_matr[0][0] + conf_matr[1][0])
recall = (conf_matr[0][0]) / (conf_matr[0][0] + conf_matr[0][1])

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

# e) dio zadatka
plt.figure()
plt.scatter(X_test[y_test == y_test_pred, 0], X_test[y_test == y_test_pred, 1], color='green', label='Correct classification')
plt.scatter(X_test[y_test != y_test_pred, 0], X_test[y_test != y_test_pred, 1], color='black', label='Wrond classification')
plt.legend()
plt.title("Test Data by Class")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()



