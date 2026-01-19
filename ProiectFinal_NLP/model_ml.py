import numpy as np

class CustomNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_log_prior = {}
        self.feature_log_prob = {}

        for c in self.classes:
            X_c = X[y == c]

            # FIX: convert to ndarray
            word_count = np.asarray(X_c.sum(axis=0)).ravel() + self.alpha
            total_count = word_count.sum()

            self.feature_log_prob[c] = np.log(word_count / total_count)
            self.class_log_prior[c] = np.log(X_c.shape[0] / X.shape[0])

    def predict(self, X):
        predictions = []

        for i in range(X.shape[0]):
            scores = {}
            for c in self.classes:
                scores[c] = (
                    self.class_log_prior[c]
                    + X[i].dot(self.feature_log_prob[c])
                )
            predictions.append(max(scores, key=scores.get))

        return np.array(predictions)
