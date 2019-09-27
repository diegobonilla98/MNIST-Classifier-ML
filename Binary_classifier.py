from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mnist = datasets.load_digits()

# 1797 numeros
X, y = mnist["data"], mnist["target"]

# ENSEÑAR NUMEROS
# -----------------------------------------------------
some_idx = int(np.random.rand(1)[0] * 1797)

some_digit = X[some_idx]
some_digit_image = some_digit.reshape(8, 8)

print("El numero es: ", y[some_idx])

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


X_train, X_test, y_train, y_test = X[:1540], X[1540:], y[:1540], y[1540:]

shuffle_index = np.random.permutation(1540)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# BINARY CLASSIFIER
# -----------------------------------------------------
# Creamos los vectores de clasificacion del numero 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
print("Entrenando...")
sgd_clf.fit(X_train, y_train_5)

print("Resultado: ", sgd_clf.predict([some_digit]))

# EVALUACION DE MODELO
# -----------------------------------------------------
cross_val = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# K-fold cross-validation devuelve las predicciones para cada instancia
# confusion matrix devuelve los false/true positives/negatives
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print("\n\nConfusion Matrix:")
print(confusion_matrix(y_train_5, y_train_pred))

# calculamos precision y recall del modelo
print("Precision: ", precision_score(y_train_5, y_train_pred))
print("Recall: ", recall_score(y_train_5, y_train_pred))
# f1 score
print("F1 score: ", f1_score(y_train_5, y_train_pred))

# ya que no te deja cambiar el threshold puedes cambiarlo manualmente
y_scores = abs(sgd_clf.decision_function([some_digit]))
print("Y scores: ", y_scores)
threshold = 2000
y_some_digit_pred = (y_scores > threshold)

y_scores_2 = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_2)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
