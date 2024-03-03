import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        # 初始化模型，使用平均值
        initial_prediction = np.mean(y)
        self.models.append(lambda x: np.full_like(x, initial_prediction))

        # 训练弱学习器
        for _ in range(self.n_estimators):
            # 计算残差
            residuals = y - self.predict(X)

            # 拟合弱学习器
            weak_learner = DecisionTreeRegressor(max_depth=3)
            weak_learner.fit(X, residuals)

            # 更新模型
            self.models.append(weak_learner.predict)

        return self

    def predict(self, X):
        # 组合所有弱学习器的输出
        predictions = np.sum(model(X) for model in self.models[1:])
        return self.models[0](X) + self.learning_rate * predictions

# 示例使用
# 创建模拟数据
np.random.seed(42)
X_train = np.random.rand(100, 1)
y_train = 3 * X_train.squeeze() + np.random.randn(100)  # 添加噪声

# 创建并训练梯度提升机模型
gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)

# 预测新数据
X_new = np.array([[0.8]])
prediction = gbm.predict(X_new)
print("Prediction:", prediction)
