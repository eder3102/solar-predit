import numpy as np
from sklearn.linear_model import LinearRegression

class Calibration:
    def __init__(self):
        self.calibrator = LinearRegression()
    
    def fit(self, preds, targets):
        """在验证集上拟合校准参数"""
        self.calibrator.fit(preds.reshape(-1,1), targets)
    
    def apply(self, preds):
        """应用校准"""
        return self.calibrator.predict(preds.reshape(-1,1)) 