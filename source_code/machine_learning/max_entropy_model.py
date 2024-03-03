import numpy as np

class MaxEntModel:
    def __init__(self, features, max_iter=100):
        self.features = features
        self.max_iter = max_iter
        self.weights = None

    def fit(self, X, Y, tolerance=1e-5):
        num_instances, num_features = X.shape
        num_classes = np.max(Y) + 1

        # 初始化权重
        self.weights = np.zeros((num_classes, num_features))

        # 计算特征函数在训练数据上的期望
        empirical_counts = np.zeros((num_classes, num_features))
        for i in range(num_instances):
            for j in range(num_features):
                empirical_counts[Y[i], j] += X[i, j]

        # 优化权重
        for _ in range(self.max_iter):
            for i in range(num_instances):
                xi = X[i, :]
                prob_dist = self._calculate_prob_dist(xi)
                for j in range(num_features):
                    for c in range(num_classes):
                        empirical_count = empirical_counts[c, j]
                        model_count = np.sum(prob_dist[k] * xi[j] for k in range(num_classes))
                        delta = (1.0 / num_instances) * np.log(empirical_count / model_count)
                        self.weights[c, j] += delta

            # 检查收敛
            if np.max(np.abs(delta)) < tolerance:
                break

    def _calculate_prob_dist(self, xi):
        scores = np.dot(self.weights, xi)
        exp_scores = np.exp(scores - np.max(scores))  # 减去最大值，防止指数爆炸
        prob_dist = exp_scores / np.sum(exp_scores)
        return prob_dist

    def predict(self, X):
        num_instances, _ = X.shape
        predictions = np.zeros(num_instances, dtype=int)

        for i in range(num_instances):
            prob_dist = self._calculate_prob_dist(X[i, :])
            predictions[i] = np.argmax(prob_dist)

        return predictions

# 示例用法
if __name__ == "__main__":
    # 训练数据
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_train = np.array([0, 1, 1, 0])

    # 特征函数
    def feature_function_1(x, y):
        return x

    def feature_function_2(x, y):
        return y

    features = [feature_function_1, feature_function_2]

    # 创建和训练最大熵模型
    maxent_model = MaxEntModel(features)
    maxent_model.fit(X_train, Y_train)

    # 测试数据
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # 预测
    predictions = maxent_model.predict(X_test)
    print("Predictions:", predictions)
