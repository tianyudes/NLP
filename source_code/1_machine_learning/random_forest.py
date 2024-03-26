import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return {'value': max(set(y), key=y.count)}

        feature, threshold = self._find_best_split(X, y)

        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        num_features = X.shape[1]
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            unique_values = set(X[:, feature])
            for threshold in unique_values:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                gini = self._calculate_gini(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, left_labels, right_labels):
        left_size = len(left_labels)
        right_size = len(right_labels)
        total_size = left_size + right_size

        gini_left = 1 - sum((np.sum(left_labels == label) / left_size) ** 2 for label in set(left_labels))
        gini_right = 1 - sum((np.sum(right_labels == label) / right_size) ** 2 for label in set(right_labels))

        weighted_gini = (left_size / total_size) * gini_left + (right_size / total_size) * gini_right

        return weighted_gini

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self._traverse_tree(self.tree, sample))
        return predictions

    def _traverse_tree(self, node, sample):
        if 'value' in node:
            return node['value']
        elif sample[node['feature']] <= node['threshold']:
            return self._traverse_tree(node['left'], sample)
        else:
            return self._traverse_tree(node['right'], sample)


class RandomForest:
    def __init__(self, n_trees, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(predictions[:, i]).argmax() for i in range(predictions.shape[1])])

# 示例用法
if __name__ == "__main__":
    # 样本特征
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    # 样本标签
    y_train = np.array([0, 0, 1, 1])

    # 创建并训练随机森林
    rf_classifier = RandomForest(n_trees=3, max_depth=2)
    rf_classifier.fit(X_train, y_train)

    # 新样本预测
    X_test = np.array([[2, 3], [4, 5]])
    predictions = rf_classifier.predict(X_test)

    print("Predictions:", predictions)
