from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mnist = datasets.load_digits()

# 1797 numeros
X, y = mnist["data"], mnist["target"]
X_prueba = np.array([0, 0, 2, 2, 2, 2, 0, 0, 0, 5, 15, 15, 15, 15, 5, 0, 2, 15, 5, 2, 2, 5, 15, 2, 2, 15, 5, 0, 0, 2, 15, 2, 0,
            5, 15, 5, 2, 5, 15, 2, 0, 0, 5, 15, 15, 15, 5, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# ENSEÑAR NUMEROS
# -----------------------------------------------------
some_idx = int(np.random.rand(1)[0] * 1797)

some_digit = X[some_idx]
some_digit_image = some_digit.reshape(8, 8)
some_digit_image_prueba = X_prueba.reshape(8, 8)

print("El numero es: ", y[some_idx])

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
plt.imshow(some_digit_image_prueba, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

X_train, X_test, y_train, y_test = X[:1540], X[1540:], y[:1540], y[1540:]

shuffle_index = np.random.permutation(1540)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# ONE VS ONE CLASSIFIER
# -----------------------------------------------------
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print("Prediccion OvO: ", ovo_clf.predict([X_prueba]))
print("Estimaciones:", len(ovo_clf.estimators_))

# RANDOM FOREST CLASSIFIER
# -----------------------------------------------------
forest_clf = RandomForestClassifier(n_estimators=10)
forest_clf.fit(X_train, y_train)
print("\nPrediccion Forest: ", forest_clf.predict([X_prueba]))
print("Estimaciones Forest: ", forest_clf.predict_proba([X_prueba]))

# evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print("Cross-val score: ", cross_val_score(ovo_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
