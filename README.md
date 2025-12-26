Here’s a fully **humanized, polished, and professional version** of your README, keeping all your results, explanations, and experiment insights clear and concise:

---

# **K-D Tree for k-Nearest Neighbors (kNN)**

## **Executive Summary**

This project implements a **K-D tree–accelerated k-Nearest Neighbors (kNN) classifier** and evaluates its performance across datasets with varying sizes and dimensionalities.

Experiments show that while k-d trees **do not provide speed advantages for small datasets** and **slow down in high-dimensional spaces** due to the curse of dimensionality, they **significantly accelerate queries on large, low-dimensional datasets**, achieving over a **12× reduction in query time** compared to brute-force kNN.

These results demonstrate the importance of choosing data structures **based on the dataset characteristics** rather than assuming universal performance improvements.

---

## **Objective**

The goal of this project is to:

1. **Accelerate kNN classification** using a k-d tree.
2. **Compare performance** against brute-force kNN and scikit-learn’s optimized kNN in terms of **accuracy** and **query latency**.

---

## **Methodology**

1. **KD-Tree Construction**

   * Built a **balanced k-d tree** by recursively splitting the dataset along alternating feature axes.
   * At each level, points are sorted along the chosen axis, and the **median point** is used as the splitting node.

2. **Querying kNN**

   * For each test sample, the tree is traversed recursively.
   * Branches are **pruned** if they cannot contain closer points than the current k nearest neighbors.
   * Final class prediction is made via **majority voting**.

3. **Baseline Methods**

   * Implemented **brute-force kNN** for reference.
   * Used **scikit-learn kNN** as an additional benchmark.

---

## **Datasets**

| Dataset   | Type                   | Samples | Features | Classes |
| --------- | ---------------------- | ------- | -------- | ------- |
| Iris      | Small, real-world      | 150     | 4        | 3       |
| Digits    | High-dimensional, real | 63,000  | 64       | 10      |
| Synthetic | Large, low-dimensional | 50,000  | 2        | 2       |

> The synthetic dataset was generated specifically to test scalability on large low-dimensional datasets.

---

## **Evaluation Metrics**

* **Accuracy**: Correct class predictions / total samples
* **Query Latency**: Time taken to predict all test samples

---

## **Experimental Results**

### **1. Small Dataset Baseline (Iris)**

* **Dataset:** 150 samples, 4 features
* **Observation:** Brute-force kNN is faster than KD-Tree due to minimal overhead on small datasets.
* **Accuracy:** 100% for all methods

### **2. High-Dimensional Stress Test (Digits)**

* **Dataset:** 62,850 training samples, 64 features
* **Observation:** KD-Tree is slower than brute-force due to ineffective pruning in high dimensions (curse of dimensionality).
* **Accuracy:** 98.33% for all methods

### **3. Low-Dimensional Scalability Test (Synthetic 2D)**

* **Dataset:** 35,000 training samples, 2 features
* **Observation:** KD-Tree significantly reduces distance computations, achieving **12× speedup** compared to brute-force.
* **Accuracy:** 99.69% for all methods

---

### **Benchmark Table**

| Dataset   | Dim | Samples | Method      | Time (s)  | Speedup vs Brute | Accuracy |
| --------- | --- | ------- | ----------- | --------- | ---------------- | -------- |
| Iris      | 4   | 150     | Brute KNN   | 0.000429  | 1.0×             | 1.0000   |
| Iris      | 4   | 150     | KD-Tree KNN | 0.003356  | 0.13×            | 1.0000   |
| Digits    | 64  | 62,850  | Brute KNN   | 10.741018 | 1.0×             | 0.9833   |
| Digits    | 64  | 62,850  | KD-Tree KNN | 87.591415 | 0.12×            | 0.9833   |
| Synthetic | 2   | 50,000  | Brute KNN   | 14.829358 | 1.0×             | 0.9969   |
| Synthetic | 2   | 50,000  | KD-Tree KNN | 1.207038  | 12.29×           | 0.9969   |

> **Note:** Accuracy remained consistent across all methods, confirming that differences in execution time are due to algorithmic efficiency rather than predictive quality.

---

## **Scalability Analysis**

* KD-Trees **scale well** with **large low-dimensional datasets**, as demonstrated by the synthetic 2D dataset.
* In **high-dimensional datasets**, KD-Trees perform worse than brute-force due to the **curse of dimensionality**.
* Small datasets like Iris show **KD-Tree overhead outweighs benefits**.

> **Theory Note:** Querying a balanced KD-Tree has **O(log N)** complexity in low dimensions but approaches **O(N)** in high dimensions due to reduced pruning efficiency.

---

## **Discussion**

* **KD-Tree efficiency** depends strongly on **dataset size** and **dimensionality**.
* For **small datasets**, brute-force is faster due to minimal overhead.
* For **high-dimensional datasets**, KD-Tree becomes slower due to ineffective pruning.
* For **large low-dimensional datasets**, KD-Tree reduces the number of distance computations, achieving substantial speedups.

---

## **Conclusion**

* KD-Trees are highly effective for **large, low-dimensional datasets**.
* Brute-force kNN is still preferable for **small or high-dimensional datasets**.
* Choosing the **right data structure** based on dataset characteristics is key to achieving optimal performance.

---

## **How to Run**

```bash
python benchmark_iris.py       # Small Iris dataset
python benchmark_digits.py     # High-dimensional Digits dataset
python benchmark_synthetic.py  # Large low-dimensional synthetic dataset
```
