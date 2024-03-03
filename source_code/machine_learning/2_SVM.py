import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 初始化权重和偏差
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # 梯度下降训练
        for epoch in range(self.epochs):
            for i, x in enumerate(X):
                condition = y[i] * (np.dot(x, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * 1 / self.epochs * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * 1 / self.epochs * self.weights - np.dot(x, y[i]))
                    self.bias -= self.learning_rate * y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

# 示例用法
if __name__ == "__main__":
    # 创建一些线性可分的数据
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([1, 1, -1, -1])

    # 创建SVM实例并训练模型
    svm_model = SVM()
    svm_model.fit(X_train, y_train)

    # 预测新样本
    X_test = np.array([[5, 6], [1, 2]])
    predictions = svm_model.predict(X_test)

    print("Predictions:", predictions)
