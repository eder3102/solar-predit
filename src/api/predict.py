# 预测服务（内存优化版）
import torch
from models import EnsembleModel
import numpy as np
from collections import deque

class PredictionService:
    def __init__(self):
        # 延迟加载模型
        self.models_loaded = False
        self.model = None
        
    def load_models(self):
        """按需加载模型以节省内存"""
        if not self.models_loaded:
            self.model = EnsembleModel.load_from_checkpoint(
                'models/ensemble.ckpt',
                map_location='cpu'  # 强制使用CPU
            )
            self.model.eval()
            self.models_loaded = True
            
    def predict(self, input_data):
        self.load_models()
        with torch.no_grad():
            # 启用内存优化模式
            with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度
                return self.model(input_data).numpy()

# 添加批处理支持
class BatchPredictor:
    def __init__(self, max_batch=64):
        self.buffer = []
        self.max_batch = max_batch

    def add_request(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= self.max_batch:
            return self.process_batch()
        return None

    @memory_safe
    def process_batch(self):
        batch = torch.stack(self.buffer)
        with torch.cuda.amp.autocast():
            result = self.model(batch)
        self.buffer = []
        return result 

# 添加预测监控
class PredictionMonitor:
    def __init__(self, window_size=1000):
        self.errors = deque(maxlen=window_size)
    
    def update(self, pred, actual):
        error = np.abs(pred - actual).mean()
        self.errors.append(error)
    
    def check_anomaly(self):
        if len(self.errors) < 100:
            return False
        return np.mean(self.errors[-100:]) > 0.15 