import numpy as np
import heapq


class KDNode:
    def __init__(self, point, label, axis):
        self.point = point
        self.label = label
        self.axis = axis
        self.left = None
        self.right = None


def build_kd_tree(points, labels, depth=0):     # function to build KD-Tree
    if points is None or len(points) == 0:
        return None

    k = points.shape[1]
    axis = depth % k


    idx = points[:, axis].argsort()
    points = points[idx]
    labels = labels[idx]

    mid = len(points) // 2


    node = KDNode(points[mid], labels[mid], axis)
    node.left = build_kd_tree(points[:mid], labels[:mid], depth + 1)
    node.right = build_kd_tree(points[mid + 1:], labels[mid + 1:], depth + 1)

    return node


def euclidean(a, b): # euclidean distance 
    diff = a - b
    return np.linalg.norm(diff)


def kd_knn_search(node, target, k, heap, depth=0): # KD-Tree kNN search function
    if node is None:
        return

    dist = euclidean(target, node.point)

    if len(heap) < k:
        heapq.heappush(heap, (-dist, node.label))
    else:
        max_dist = -heap[0][0]
        
        if dist < max_dist:
            heapq.heappushpop(heap, (-dist, node.label))

    axis = node.axis
    diff = target[axis] - node.point[axis]

    if diff < 0:             # traverse left first
        near = node.left
        far = node.right
    else:
        near = node.right
        far = node.left

    kd_knn_search(near, target, k, heap, depth + 1)

    if len(heap) < k:           # check if we have k neighbors yet
        check_other_side = True
    else:
        max_dist = -heap[0][0]
        check_other_side = abs(diff) < max_dist

    if check_other_side:  # traverse the other side if necessary
        kd_knn_search(far, target, k, heap, depth + 1)


def kd_predict(root, X_test, k):  # KD-Tree kNN prediction function

    preds = []

    for x in X_test:
        heap = []
        kd_knn_search(root, x, k, heap)

        labels = [label for (_, label) in heap]
        preds.append(max(set(labels), key=labels.count))

    return np.array(preds)


def brute_knn(X_train, y_train, X_test, k):   # Brute-force kNN function
    preds = []

    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(dists)[:k]

        labels = y_train[idx]

        preds.append(max(set(labels), key=list(labels).count))

    return np.array(preds)
