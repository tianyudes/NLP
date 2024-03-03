import numpy as np
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y, depth=0):
        unique_classes = list(set(y))

        # 如果所有样本属于同一类别，创建叶节点
        if len(unique_classes) == 1:
            return {'class': unique_classes[0]}

        # 如果达到最大深度，创建叶节点，类别为样本中最频繁的类别
        if self.max_depth is not None and depth == self.max_depth:
            return {'class': max(set(y), key=y.count)}

        # 选择最佳特征和切分点
        best_feature, best_value = self.choose_best_split(X, y)

        # 如果无法找到最佳特征，创建叶节点，类别为样本中最频繁的类别
        if best_feature is None:
            return {'class': max(set(y), key=y.count)}

        # 递归构建左右子树
        left_indices = X[:, best_feature] <= best_value
        right_indices = ~left_indices

        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': best_feature, 'split_value': best_value,
                'left': left_subtree, 'right': right_subtree}

    def choose_best_split(self, X, y):
        # 根据信息增益或其他标准选择最佳特征和切分点
        # 在这里，简化为选择第一个可用特征和其中值作为切分点
        if X.shape[1] > 0:
            return 0, np.median(X[:, 0])
        else:
            return None, None

    def predict_single(self, tree, sample):
        if 'class' in tree:
            return tree['class']

        if sample[tree['feature_index']] <= tree['split_value']:
            return self.predict_single(tree['left'], sample)
        else:
            return self.predict_single(tree['right'], sample)

    def predict(self, X):
        return [self.predict_single(self.tree, sample) for sample in X]

# 示例用法
if __name__ == "__main__":
    # 创建一些简单的二分类数据
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    # 创建决策树实例并训练模型
    dt_model = DecisionTree(max_depth=2)
    dt_model.tree = dt_model.fit(X_train, y_train)

    # 预测新样本
    X_test = np.array([[5, 6], [1, 2]])
    predictions = dt_model.predict(X_test)

    print("Predictions:", predictions)
