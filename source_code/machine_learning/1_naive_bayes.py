class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = {}  
        self.feature_probabilities = {}  

    def calculate_class_probabilities(self, labels):
        total_samples = len(labels)
        for label in set(labels):
            self.class_probabilities[label] = labels.count(label) / total_samples

    def calculate_feature_probabilities(self, features, labels):
        unique_labels = set(labels)
        for label in unique_labels:
            label_indices = [i for i, x in enumerate(labels) if x == label]
            label_features = [features[i] for i in label_indices]

            for feature_index in range(len(label_features[0])):
                feature_values = [sample[feature_index] for sample in label_features]
                unique_values = set(feature_values)

                if label not in self.feature_probabilities:
                    self.feature_probabilities[label] = {}

                if feature_index not in self.feature_probabilities[label]:
                    self.feature_probabilities[label][feature_index] = {}

                for value in unique_values:
                    count = feature_values.count(value)
                    probability = count / len(label_features)
                    self.feature_probabilities[label][feature_index][value] = probability

    def predict(self, sample):
        max_probability = -1
        predicted_class = None

        for label, class_probability in self.class_probabilities.items():
            feature_probability = 1.0

            for feature_index, feature_value in enumerate(sample):
                if feature_index in self.feature_probabilities[label] and feature_value in self.feature_probabilities[label][feature_index]:
                    feature_probability *= self.feature_probabilities[label][feature_index][feature_value]

            probability = class_probability * feature_probability

            if probability > max_probability:
                max_probability = probability
                predicted_class = label

        return predicted_class



if __name__ == "__main__":
    training_data = [
        [1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],
        [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'],
        [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'],
        [3, 'M'], [3, 'L'], [3, 'L']
    ]

    labels = ['N', 'N', 'Y', 'Y', 'N', 'N', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N']

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.calculate_class_probabilities(labels)
    nb_classifier.calculate_feature_probabilities(training_data, labels)

    new_sample = [2, 'S']
    prediction = nb_classifier.predict(new_sample)
    print(f"Prediction for {new_sample}: {prediction}")
