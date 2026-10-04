import numpy as np
from mapie.classification import SplitConformalClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class MockEst(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = np.array([0, 1])
    def fit(self, X, y):
        return self
    def predict_proba(self, X):
        return np.array([[0.1, 0.9], [0.8, 0.2]])

clf = SplitConformalClassifier(estimator=MockEst(), confidence_level=0.9, prefit=True)
clf.conformalize(np.zeros((2,1)), np.array([1, 0]))
res = clf.predict(np.zeros((2,1)))
print("Shape:", res.shape, "Type:", type(res))
print("Result:\n", res)
