import torch
from torch.cuda.amp import autocast

class OptimizedPredictionService:
    def __init__(self):
        self.models = {
            'filter_net': FilterNet().cuda(),
            'bilstm': BiLSTM().cuda(),
            'xgb': XGBoostModel()
        }
        
    @torch.inference_mode()
    def predict(self, inputs):
        with autocast():
            # 并行执行三个模型
            f_future = torch.jit.fork(self.models['filter_net'], inputs)
            b_future = torch.jit.fork(self.models['bilstm'], inputs)
            x_result = self.models['xgb'].predict(inputs.cpu())
            
            f_result = torch.jit.wait(f_future)
            b_result = torch.jit.wait(b_future)
            
        # 动态融合
        return self.fuse_results(f_result, b_result, x_result) 