import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from kd_tree import build_kd_tree, kd_predict, brute_knn


digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# increase dataset size to test scalability
X_train = np.repeat(X_train, 50, axis=0)
y_train = np.repeat(y_train, 50, axis=0)

k = 5



#KD-Tree KNN
root = build_kd_tree(X_train, y_train)

start = time.perf_counter()
kd_preds = kd_predict(root, X_test, k)
kd_time = time.perf_counter() - start
kd_acc = accuracy_score(y_test, kd_preds)




#Brute-force KNN
start = time.perf_counter()
brute_preds = brute_knn(X_train, y_train, X_test, k)
brute_time = time.perf_counter() - start
brute_acc = accuracy_score(y_test, brute_preds)

#Scikit-learn KNN
model = KNeighborsClassifier(n_neighbors=k)

start = time.perf_counter()
model.fit(X_train, y_train)


sk_preds = model.predict(X_test)
sk_time = time.perf_counter() - start
sk_acc = accuracy_score(y_test, sk_preds)


#Speedup ratios

kd_vs_brute = brute_time / kd_time if kd_time > 0 else float("inf")
kd_vs_sklearn = sk_time / kd_time if kd_time > 0 else float("inf")


#Results
print("\nExperiment: kNN on High-Dimensional Dataset (Digits)")
print(f"Features: {X.shape[1]}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"k = {k}\n")


print("Method\t\tAccuracy\tTime (s)")
print(f"Brute KNN\t{brute_acc:.4f}\t\t{brute_time:.6f}")
print(f"KD-Tree KNN\t{kd_acc:.4f}\t\t{kd_time:.6f}")
print(f"Sklearn KNN\t{sk_acc:.4f}\t\t{sk_time:.6f}")


print("\nSpeedup Ratios:")
print(f"KD-Tree vs Brute-force: {kd_vs_brute:.2f}x")
print(f"KD-Tree vs Sklearn KNN: {kd_vs_sklearn:.2f}x")

# Compares brute-force, KD-Tree, and scikit-learn kNN on high-dimensional Digits dataset, KD-Tree slower due to high dimensionality and tree overhead.
# KD-Tree much slower due to high dimensionality