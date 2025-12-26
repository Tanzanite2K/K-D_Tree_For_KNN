import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from kd_tree import build_kd_tree, kd_predict, brute_knn

np.random.seed(42)

X = np.random.rand(50000, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k = 5



#KD-Tree KNN
root = build_kd_tree(X_train, y_train)

start = time.perf_counter()
kd_preds = kd_predict(root, X_test, k)
kd_time = time.perf_counter() - start
kd_acc = accuracy_score(y_test, kd_preds)



#Brute-force kNN
start = time.perf_counter()
brute_preds = brute_knn(X_train, y_train, X_test, k)

brute_time = time.perf_counter() - start
brute_acc = accuracy_score(y_test, brute_preds)


#Scikit-learn kN
model = KNeighborsClassifier(n_neighbors=k)

start = time.perf_counter()
model.fit(X_train, y_train)

sk_preds = model.predict(X_test)
sk_time = time.perf_counter() - start
sk_acc = accuracy_score(y_test, sk_preds)


#Speedup Ratio


kd_vs_brute = brute_time / kd_time if kd_time > 0 else float("inf")
kd_vs_sklearn = sk_time / kd_time if kd_time > 0 else float("inf")


#Results
print("\nExperiment: kNN on Low-Dimensional Large Dataset (2D)")
print(f"Total samples: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")
print(f"k = {k}\n")

print("Method\t\tAccuracy\tTime (s)")

print(f"Brute kNN\t{brute_acc:.4f}\t\t{brute_time:.6f}")
print(f"KD-Tree kNN\t{kd_acc:.4f}\t\t{kd_time:.6f}")
print(f"Sklearn kNN\t{sk_acc:.4f}\t\t{sk_time:.6f}")

print("\nSpeedup Ratios:")

print(f"KD-Tree vs Brute-force: {kd_vs_brute:.2f}x faster")
print(f"KD-Tree vs Sklearn kNN: {kd_vs_sklearn:.2f}x faster")

# Compares brute-force, KD-Tree, and scikit-learn kNN on a large low-dimensional (2D) dataset, KD-Tree gives significant speedup due to effective spatial pruning.
# KD-Tree much faster than brute-force, slower than optimized scikit-learn