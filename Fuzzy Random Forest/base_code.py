import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def trapezoidal_membership(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    
def fuzzy_entropy(class_weights):
    total = sum(class_weights.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for w in class_weights.values():
        if w > 0:
            p = w / total
            entropy -= p * np.log2(p)
    return entropy

class FuzzyNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.feature = None
        self.membership_funcs = {}   # baixo, medio, alto
        self.children = {}
        self.class_distribution = {}

class FuzzyDecisionTree:
    def __init__(self, max_depth=5, min_entropy=0.01, m_features=None):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.m_features = m_features
        self.root = None

    def fit(self, X, y, weights):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, weights, depth=0)

    def _build_tree(self, X, y, weights, depth):
        node = FuzzyNode()

        # Soma fuzzy por classe
        class_weights = defaultdict(float)
        for label, w in zip(y, weights):
            class_weights[label] += w

        entropy = fuzzy_entropy(class_weights)

        # Critério de parada
        if depth >= self.max_depth or entropy < self.min_entropy:
            node.is_leaf = True
            total = sum(class_weights.values())
            node.class_distribution = {
                c: w / total for c, w in class_weights.items()
            }
            return node
        
        n_features = X.shape[1]
        features = np.random.choice(
            n_features,
            self.m_features or int(np.sqrt(n_features)),
            replace=False
        )

        best_feature = features[0]
        node.feature = best_feature

        values = X[:, best_feature].reshape(-1, 1)

        K = min(3, len(np.unique(values)))
        kmeans = KMeans(n_clusters=K, n_init=10)
        kmeans.fit(values)

        centers = np.sort(kmeans.cluster_centers_.flatten())

        if K == 3:
            a, b, c = centers
            m1 = (a + b) / 2
            m2 = (b + c) / 2

            node.membership_funcs = {
                "baixo":  (a - 1e-5, a, m1, b),
                "medio":  (a, m1, m2, c),
                "alto":   (b, m2, c, c + 1e-5)
            }
        else:
            a, b = centers
            m = (a + b) / 2
            node.membership_funcs = {
                "baixo": (a - 1e-5, a, m, b),
                "alto":  (a, m, b, b + 1e-5)
            }

        for branch, params in node.membership_funcs.items():
            new_weights = []
            for xi, wi in zip(X[:, best_feature], weights):
                mu = trapezoidal_membership(xi, *params)
                new_weights.append(wi * mu)

            new_weights = np.array(new_weights)

            if new_weights.sum() > 0:
                node.children[branch] = self._build_tree(
                    X, y, new_weights, depth + 1
                )

        return node
    
    def predict_proba(self, x):
        return self._predict_node(self.root, x, weight=1.0)

    def _predict_node(self, node, x, weight):
        if node.is_leaf:
            return {
                c: weight * p for c, p in node.class_distribution.items()
            }

        results = defaultdict(float)
        value = x[node.feature]

        for branch, params in node.membership_funcs.items():
            mu = trapezoidal_membership(value, *params)
            if mu > 0 and branch in node.children:
                child_res = self._predict_node(
                    node.children[branch], x, weight * mu
                )
                for c, v in child_res.items():
                    results[c] += v

        return results

class FuzzyRandomForest:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []


    def fit(self, X, y):
        n = len(X)

        for _ in range(self.n_estimators):
            indices = np.random.choice(n, n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            weights = np.ones(len(X_boot))

            tree = FuzzyDecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot, weights)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []

        for x in X:
            total = defaultdict(float)

            for tree in self.trees:
                probs = tree.predict_proba(x)
                for c, v in probs.items():
                    total[c] += v

            predictions.append(max(total, key=total.get))

        return np.array(predictions)
        