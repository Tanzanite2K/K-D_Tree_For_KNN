import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from kd_tree import build_kd_tree, kd_predict, brute_knn


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k = 5



# KD-Tree KNN
root = build_kd_tree(X_train, y_train)

start = time.perf_counter()
kd_preds = kd_predict(root, X_test, k)   
kd_time = time.perf_counter() - start
kd_acc = accuracy_score(y_test, kd_preds)

# Brute-force KNN
start = time.perf_counter()
brute_preds = brute_knn(X_train, y_train, X_test, k)
brute_time = time.perf_counter() - start
brute_acc = accuracy_score(y_test, brute_preds)

#


# Scikit-learn kNN
model = KNeighborsClassifier(n_neighbors=k)

start = time.perf_counter()
model.fit(X_train, y_train)

sk_preds = model.predict(X_test)
sk_time = time.perf_counter() - start
sk_acc = accuracy_score(y_test, sk_preds)



# Speedup Ratios
kd_vs_brute = brute_time / kd_time if kd_time > 0 else float("inf")
kd_vs_sklearn = sk_time / kd_time if kd_time > 0 else float("inf")


# Results
print("\nkNN Classification on Iris Dataset")
print(f"k = {k}, Train size = {len(X_train)}, Test size = {len(X_test)}\n")
print("Method\t\tAccuracy\tTime (s)")

print(f"Brute KNN\t{brute_acc:.4f}\t\t{brute_time:.6f}")
print(f"KD-Tree KNN\t{kd_acc:.4f}\t\t{kd_time:.6f}")
print(f"Sklearn KNN\t{sk_acc:.4f}\t\t{sk_time:.6f}")


print("\nSpeedup Ratios:")

print(f"KD-Tree vs Brute-force: {kd_vs_brute:.2f}x faster")
print(f"KD-Tree vs Sklearn KNN: {kd_vs_sklearn:.2f}x faster")

# Compares brute-force, KD-Tree, and scikit-learn kNN on small Iris dataset, KD-Tree slightly slower than Brute-force KNN due to overhead.
# KD-Tree slightly slower due to small dataset