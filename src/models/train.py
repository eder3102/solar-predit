# 训练主程序（分阶段训练设计）
import argparse
from models import FilterNet, BiLSTMModel, DynamicWeightLayer
from data_processing import FeatureProcessor
from xgboost import XGBRegressor
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast

def train_xgboost(features, labels):
    """第一阶段：训练XGBoost模型"""
    model = XGBRegressor(
        n_estimators=100,  # 根据内存调整树的数量
        max_depth=6,      # 限制树深度
        tree_method='hist',# 内存优化模式
        enable_categorical=True
    )
    model.fit(features, labels)
    return model

def train_neural_net(model_class, train_loader, epochs=50):
    """第二阶段：训练神经网络模型"""
    model = model_class(input_dim=91, hidden_dim=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 内存优化配置
    torch.backends.cudnn.benchmark = True  # 启用CUDA优化（如果可用）
    torch.set_float32_matmul_precision('medium')  # 降低计算精度节省内存
    
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for batch in train_loader:
            with autocast():
                outputs = model(batch)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model

if __name__ == "__main__":
    # 配置内存优化参数
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.set_num_threads(2)  # 限制CPU线程数
    
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1,2], required=True)
    args = parser.parse_args()
    
    # 特征处理
    processor = FeatureProcessor()
    features, labels = processor.load_data()
    
    if args.phase == 1:
        xgb_model = train_xgboost(features, labels)
        xgb_model.save_model('models/xgboost.json')
    elif args.phase == 2:
        train_loader = create_data_loader(features, labels, batch_size=32)
        filter_net = train_neural_net(FilterNet, train_loader)
        bilstm = train_neural_net(BiLSTMModel, train_loader) 