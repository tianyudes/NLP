import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Example usage:
# X_train and y_train are training data
X_train = np.array([[1, 2], [2, 3], [3, 1], [1, 1]])
y_train = np.array([0, 1, 0, 1])

# Initialize and train the model
knn_model = KNN(k=3)
knn_model.fit(X_train, y_train)

# Test the model
X_test = np.array([[2, 2], [1, 3]])
predictions = knn_model.predict(X_test)
print("Predictions:", predictions)
