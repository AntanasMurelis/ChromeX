from collections import defaultdict
import numpy as np
from sklearn.base import BaseEstimator


class MultiClassifier():

    Classifiers = defaultdict(BaseEstimator)
    data: np.ndarray
    labels: dict
    
    def __init__(self, classifier, classifierNames):
        self.Classifiers = {classifierName: classifier for classifierName in classifierNames}   
        
    def fit(self, data, labels):
        
        for classifierName, labels in zip(self.labels.items()):
            self.Classifiers[classifierName].fit(data, labels)
          
    def predict(self, data):
        
        predictions = defaultdict(np.ndarray)
        for classifierName, classifier in self.Classifiers.items():
            predictions[classifierName] = classifier.predict(data)
        
        return predictions
    

    