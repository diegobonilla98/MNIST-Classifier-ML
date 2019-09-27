from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mnist = datasets.load_digits()

# 1797 numeros
X, y = mnist["data"], mnist["target"]

# ENSEÑAR NUMEROS
# -----------------------------------------------------
some_idx = int(np.random.rand(1)[0] * 270)

some_digit = X[some_idx]
some_digit_image = some_digit.reshape(8, 8)

print("El numero es: ", y[some_idx])

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")


X_train, X_test, y_train, y_test = X[:1540], X[1540:], y[:1540], y[1540:]

shuffle_index = np.random.permutation(1540)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# K-NEIGHBORS CLASSIFIER
# -----------------------------------------------------
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print("Predictioon: ", knn_clf.predict([some_digit]))

# evaluation
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
f1 = f1_score(y_train, y_train_knn_pred, average="macro")
print("F1 score: ", f1)

# MULTIOUTPUT CLASSIFICATION
# -----------------------------------------------------
noise_train = np.random.randint(0, 100, (len(X_train), 64))
noise_test = np.random.randint(0, 100, (len(X_test), 64))

# le metemos ruido a las imagenes y damos las que no tienen ruido para aprender
# en teoria deberia quitarle el ruido
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test

some_digit_mod = X_test_mod[some_idx]
some_digit_image_mod = some_digit_mod.reshape(8, 8)
plt.imshow(some_digit_image_mod, cmap=matplotlib.cm.binary, interpolation="nearest")

knn_clf.fit(X_test_mod, y_test_mod)
clean_digit = knn_clf.predict(X_test_mod[some_idx])

some_digit_image_mod_result = clean_digit.reshape(8, 8)
plt.imshow(some_digit_image_mod_result, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
